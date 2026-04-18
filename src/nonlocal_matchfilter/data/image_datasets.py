from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

import albucore.functions as AF
import numpy as np
import tifffile
import torch
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch.transforms import ToTensorV2
from einops import pack
from PIL import Image
from torchvision.datasets import VisionDataset

from ..isp.pipeline import (
    camera_color_correction,
    demosaic,
    gamma_correction,
    white_balance,
)
from ..isp.pipeline import unpack as cfa_unpack
from .serialize import TorchSerializedList

type AlbuTransform = BasicTransform | Compose


class ImageDegradationDataset(ABC, VisionDataset):
    def __init__(self, root: str, transforms: AlbuTransform | None = None):
        super().__init__(root=root, transforms=transforms)
        self.transforms = transforms
        self._gt_list: TorchSerializedList[Path] = None
        self._degraded_list: TorchSerializedList[Path] = None
        self._to_tensor = ToTensorV2()

    @abstractmethod
    def _read_image(self, file_name: Path) -> np.ndarray:
        pass

    @abstractmethod
    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        pass

    def __getitem__(self, index: int):
        degraded_image_path = self._degraded_list[index]
        img_degraded = self._read_image(degraded_image_path)
        metadata = self._extract_metadata(degraded_image_path)

        if self._gt_list:
            img_gt = self._read_image(self._gt_list[index])
        else:
            img_gt = None

        if self.transforms is not None:
            if img_gt is not None:
                transformed = self.transforms(image=img_degraded, gt=img_gt)
                img_degraded, img_gt = transformed["image"], transformed["gt"]
            else:
                img_degraded = self.transforms(image=img_degraded)["image"]

        img_degraded = self._to_tensor(image=img_degraded)["image"]
        if img_gt is not None:
            img_gt = self._to_tensor(image=img_gt)["image"]

        return img_degraded, img_gt if img_gt is not None else [], metadata

    @abstractmethod
    def process_image(self, img, **kwargs):
        pass

    @abstractmethod
    def canonical_name(self, **kwargs):
        pass

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, new_transforms):
        self._transforms = new_transforms
        if self.transforms is not None:
            self.transforms.add_targets({"gt": "image"})

    def __len__(self) -> int:
        return len(self._degraded_list)

    def __rmul__(self, v: int) -> torch.utils.data.ConcatDataset:
        return torch.utils.data.ConcatDataset([self] * v)


class DNDRaw(ImageDegradationDataset):
    def __init__(
        self,
        root: str,
        crops: bool = True,
        noise_map: bool = False,
        transforms: AlbuTransform | None = None,
    ):
        super().__init__(
            root=root,
            transforms=transforms,
        )
        self.info_data = np.load(
            Path(root) / "metadata_cam.npy", allow_pickle=True
        ).item()
        self.crops = crops
        self.noise_map = noise_map

        root_data = Path(root)
        root_data = root_data / "crops" if crops else root_data / "data"

        degraded_list = sorted(list(root_data.iterdir()))

        self._degraded_list = TorchSerializedList(degraded_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = tifffile.imread(file_name)
        if self.noise_map:
            scene = file_name.stem.split("_")[0]
            sigma_shot = self.info_data[scene]["sigma_shot"]
            sigma_read = self.info_data[scene]["sigma_read"]
            std_map = np.sqrt(np.maximum(sigma_shot * img + sigma_read, 0.0))
            img, _ = pack([img, std_map], "h w *")
        return img.astype(np.float32)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        scene = file_name.stem.split("_")[0]
        crop = int(file_name.stem.split("_")[1][4:]) if self.crops else 0
        return {"dataset": "DND", "scene": scene, "crop": crop}

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        scene = kwargs["scene"]
        image_data = self.info_data[scene]

        pattern = image_data["pattern"].upper()
        color_correction_matrix = image_data["ccm"]
        wb_values = image_data["awb"][:3]
        gamma = image_data["gamma"]
        max_value = 2**16 - 1

        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img_cfa = np.clip(
            cfa_unpack(img[..., :4], pattern) * max_value, 0, max_value
        ).astype(np.uint16)
        img_demosaicked = demosaic(img_cfa, pattern)
        img_demosaicked = img_demosaicked / max_value

        img_demosaicked = white_balance(img_demosaicked, wb_values)
        img_demosaicked = camera_color_correction(
            img_demosaicked, color_correction_matrix
        )
        img_demosaicked = np.clip(img_demosaicked, 0, 1)
        img_demosaicked = gamma_correction(img_demosaicked, gamma, 1.0) * 255
        img_demosaicked = np.clip(np.round(img_demosaicked), 0, 255).astype(np.uint8)
        return img_demosaicked

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['crop']:02g}"


class WildDualDnRaw(ImageDegradationDataset):
    def __init__(
        self,
        root: str,
        noise_map: bool = False,
        transforms: AlbuTransform | None = None,
    ):
        super().__init__(
            root=root,
            transforms=transforms,
        )
        self.info_data = np.load(
            Path(root) / "metadata_corrected.npy", allow_pickle=True
        ).item()
        self.noise_map = noise_map

        root_data = Path(root) / "data"

        degraded_list = sorted(list(root_data.iterdir()))

        self._degraded_list = TorchSerializedList(degraded_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = tifffile.imread(file_name)
        if self.noise_map:
            scene = file_name.stem.split("_")[1]
            sigma_shot = self.info_data[scene]["sigma_shot"]
            sigma_read = self.info_data[scene]["sigma_read"]
            std_map = np.sqrt(np.maximum(sigma_shot * img + sigma_read, 0.0))
            img, _ = pack([img, std_map], "h w *")
        return img.astype(np.float32)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        scene = file_name.stem.split("_")[1]
        return {"dataset": "WildDualDn", "scene": scene}

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        scene = kwargs["scene"]
        image_data = self.info_data[scene]

        pattern = image_data["pattern"].upper()
        color_correction_matrix = image_data["ccm"]
        wb_values = image_data["awb"][:3]
        gamma = 2.4
        max_value = 2**16 - 1

        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img_cfa = np.clip(
            cfa_unpack(img[..., :4], pattern) * max_value, 0, max_value
        ).astype(np.uint16)
        img_demosaicked = demosaic(img_cfa, pattern)
        img_demosaicked = img_demosaicked / max_value

        img_demosaicked = white_balance(img_demosaicked, wb_values)
        img_demosaicked = camera_color_correction(
            img_demosaicked, color_correction_matrix
        )
        img_demosaicked = np.clip(img_demosaicked, 0, 1)
        img_demosaicked = gamma_correction(img_demosaicked, gamma, 1.0) * 255
        img_demosaicked = np.clip(np.round(img_demosaicked), 0, 255).astype(np.uint8)
        return img_demosaicked

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}"


class GenericNoisyTestSet(ImageDegradationDataset):
    """
    Assumes 8-bit PIL-readable images and the following directory structure:
        dataset
        ├── scene1
        │   ├── 0000.png
        │   ├── 0001.png
        │   ├── 0002.png
        │   └── ...
        ├── scene2
        │   ├── 0000.png
        │   ├── 0001.png
        │   ├── 0002.png
        │   └── ...
        └── ...
    """

    IMAGE_FILE_SATURATION_VALUE = 255

    def __init__(
        self,
        root: str,
        transforms: AlbuTransform | None = None,
    ):
        super().__init__(
            root=root,
            transforms=transforms,
        )

        root_data = Path(root)
        scene_names = [scene.name for scene in root_data.iterdir()]

        degraded_list = []
        for scene in scene_names:
            image_names = [im.name for im in (root_data / scene).iterdir()]
            degraded_list.extend([root_data / scene / name for name in image_names])

        self._degraded_list = TorchSerializedList(degraded_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(file_name)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.clip(AF.to_float(np.asarray(img), 255), 0, 1)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem)
        scene = file_name.parent.stem
        dataset = file_name.parent.parent.stem
        return {"dataset": dataset, "scene": scene, "frame": frame_number}

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:04g}"


class GenericTiffTestSet(ImageDegradationDataset):
    """
    Assumes floating point TIFF images and the following directory structure:
        dataset
        ├── 0000.tiff
        ├── 0001.tiff
        ├── 0002.tiff
        └── ...
    """

    def __init__(
        self,
        root: str,
        transforms: AlbuTransform | None = None,
    ):
        super().__init__(
            root=root,
            transforms=transforms,
        )

        root_data = Path(root)

        degraded_list = list(root_data.iterdir())
        self._degraded_list = TorchSerializedList(degraded_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        return tifffile.imread(file_name).astype(np.float32)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem)
        dataset = file_name.parent.stem
        return {"dataset": dataset, "frame": frame_number}

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['frame']:04g}"


class SyntheticImageDegradationDataset(ABC, VisionDataset):
    def __init__(
        self,
        root: str,
        synthesis: Callable[..., AlbuTransform],
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(root=root, transforms=transforms)
        self.synthesis_partial = synthesis
        self.synthesis_seed = synthesis_seed
        self.load_upfront = load_upfront

        self._gt_list: TorchSerializedList[Path] = self._list_images()
        self._gt_images: TorchSerializedList[np.ndarray] = None
        if load_upfront:
            self._gt_images = TorchSerializedList(
                [self._read_image(p) for p in self._gt_list]
            )

        self._to_tensor = ToTensorV2()

    @abstractmethod
    def _list_images(self):
        pass

    @abstractmethod
    def _read_image(self, file_name: Path) -> np.ndarray:
        pass

    @abstractmethod
    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        pass

    @abstractmethod
    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        pass

    def __getitem__(self, index: int):
        image_path = self._gt_list[index]
        img_gt = (
            self._gt_images[index]
            if self.load_upfront
            else self._read_image(image_path)
        )
        metadata = self._extract_metadata(image_path)

        if self.transforms is not None:
            img_gt = self.transforms(image=img_gt)["image"]
        img_gt = self._normalize(img_gt, metadata)

        seed = self.synthesis_seed + index if self.synthesis_seed is not None else None
        synthesis = self.synthesis_partial(seed=seed)
        img_degraded = synthesis(image=img_gt)["image"]

        img_degraded = self._to_tensor(image=img_degraded)["image"]
        img_gt = self._to_tensor(image=img_gt)["image"]

        return img_degraded, img_gt, metadata

    @abstractmethod
    def process_image(self, img, metadata):
        pass

    @abstractmethod
    def canonical_name(self, metadata):
        pass

    def __len__(self) -> int:
        return len(self._gt_list)

    def __rmul__(self, v: int) -> torch.utils.data.ConcatDataset:
        return torch.utils.data.ConcatDataset([self] * v)


class SyntheticKodak24(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem[5:])
        return {"dataset": "Kodak24", "scene": "kodim", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:02g}"


class SyntheticBSD100(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem.split("_")[1])
        return {"dataset": "BSD100", "scene": "SRF_2_HR", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:03g}"


class SyntheticCBSD68(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem)
        return {"dataset": "CBSD68", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['frame']:04g}"


class SyntheticUrban100(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem.split("_")[1])
        return {"dataset": "Urban100", "scene": "0", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['frame']:03g}"


class SyntheticMcMaster(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem.split("_")[1])
        return {"dataset": "McMaster", "scene": "mcm", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:02g}"


class SyntheticSet14(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
    ):
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
        )

    def _list_images(self):
        gt_list = list(Path(self.root).iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number = int(file_name.stem.split("_")[1])
        return {"dataset": "Set14", "scene": "SRF_2_HR", "frame": frame_number}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:03g}"


class SyntheticUIBSelection(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        self.split = split
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        root_data = Path(self.root) / self.split
        gt_list = list(root_data.iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(str(file_name))
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        frame_number, original_dataset, orig_name = file_name.stem.split(
            "_", maxsplit=3
        )
        return {
            "dataset": "UIBSelection",
            "scene": original_dataset,
            "frame": int(frame_number),
        }

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}-{kwargs['frame']:04g}"


class SyntheticLSDIR(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        self.split = split
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        root_data = Path(self.root) / self.split

        if self.split == "train":
            folder_separation = [folder.name for folder in (root_data / "HR").iterdir()]
        else:
            folder_separation = [""]

        gt_list = []
        for folder in folder_separation:
            gt_list.extend(sorted(list((root_data / "HR" / folder).iterdir())))
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        img = Image.open(file_name)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        scene = file_name.stem[:7]
        return {"dataset": "LSDIR", "scene": scene}

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        return np.clip(AF.to_float(image, 255), 0, 1)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        img_processed = img.permute(1, 2, 0).detach().cpu().numpy()
        img_processed = np.clip(np.round(img_processed[..., :3] * 255), 0, 255)
        return img_processed.astype(np.uint8)

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['scene']}"


class SyntheticUIBRawSelection(SyntheticImageDegradationDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        synthesis: AlbuTransform,
        synthesis_seed: int | None = None,
        transforms: AlbuTransform | None = None,
        load_upfront: bool = False,
    ):
        self.split = split
        self.info_data = np.load(Path(root) / "metadata.npy", allow_pickle=True).item()
        super().__init__(
            root=root,
            synthesis=synthesis,
            synthesis_seed=synthesis_seed,
            transforms=transforms,
            load_upfront=load_upfront,
        )

    def _list_images(self):
        root_data = Path(self.root) / "gt" / self.split
        gt_list = list(root_data.iterdir())
        return TorchSerializedList(gt_list)

    def _read_image(self, file_name: Path) -> np.ndarray:
        return tifffile.imread(file_name)

    def _extract_metadata(self, file_name: Path) -> dict[str, any]:
        original_dataset, original_name = file_name.stem.split("_", maxsplit=1)
        return {
            "dataset": "UIBRawSelection",
            "origin": original_dataset,
            "scene": original_name,
        }

    def _normalize(self, image: np.ndarray, metadata: dict[str, any]) -> np.ndarray:
        file_name = "_".join([metadata["origin"], metadata["scene"]])
        image_data = self.info_data[file_name]

        black_level = image_data["black_level"]
        saturation_level = image_data["saturation_level"]
        image = image.astype(np.float32) - black_level
        image /= saturation_level - black_level
        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int):
        return super().__getitem__(index)

    def process_image(self, img, **kwargs):
        file_name = "_".join([kwargs["origin"], kwargs["scene"]])
        image_data = self.info_data[file_name]

        pattern = image_data["pattern"].upper()
        color_correction_matrix = image_data["ccm"]
        wb_values = image_data["awb"][:3]
        max_value = 2**16 - 1

        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img_cfa = np.clip(
            cfa_unpack(img[..., :4], pattern) * max_value, 0, max_value
        ).astype(np.uint16)
        img_demosaicked = demosaic(img_cfa, pattern)
        img_demosaicked = img_demosaicked / max_value

        img_demosaicked = white_balance(img_demosaicked, wb_values)
        img_demosaicked = camera_color_correction(
            img_demosaicked, color_correction_matrix
        )
        img_demosaicked = np.clip(img_demosaicked, 0, 1)
        img_demosaicked = gamma_correction(img_demosaicked, 2.2, 1.0) * 255
        img_demosaicked = np.clip(np.round(img_demosaicked), 0, 255).astype(np.uint8)
        return img_demosaicked

    def canonical_name(self, **kwargs):
        return f"{kwargs['dataset']}-{kwargs['origin']}-{kwargs['scene']}"
