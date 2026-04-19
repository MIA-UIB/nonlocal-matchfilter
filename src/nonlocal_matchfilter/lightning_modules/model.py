from pathlib import Path
from typing import Callable

import aim
import lightning as L
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class LitImageDenoisingModule(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: DictConfig,
    ):
        super().__init__()
        self.model = network
        self.loss_criterion = loss
        self.optimizer_partial = optimizer
        self.scheduler_config = scheduler
        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))

        self.example_input_array = torch.empty(
            1, self.model.in_channels, 128, 128, device=self.device
        )

    def forward(self, image, **kwargs):
        return self.model(image, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())

        if self.scheduler_config.get("scheduler"):
            lr_scheduler = self.scheduler_config.scheduler(optimizer=optimizer)

            lr_scheduler_dict = {"scheduler": lr_scheduler}
            if self.scheduler_config.get("extras"):
                lr_scheduler_dict.update(**self.scheduler_config.extras)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}

    def training_step(self, train_batch, batch_idx):
        noisy_img, gt_img, metadata = train_batch
        denoised_img = self.forward(noisy_img)

        denoised_img = F.hardtanh(denoised_img, 0.0, 1.0)
        gt_img = F.hardtanh(gt_img, 0.0, 1.0)

        loss = self.loss_criterion(denoised_img, gt_img)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        noisy_img, gt_img, metadata = val_batch
        denoised_img = self.forward(noisy_img)

        denoised_img = F.hardtanh(denoised_img, 0.0, 1.0)
        gt_img = F.hardtanh(gt_img, 0.0, 1.0)

        loss = self.loss_criterion(denoised_img, gt_img)
        self.log("val_loss", loss)
        self.psnr(denoised_img, gt_img)
        self.log("val_psnr", self.psnr)
        self.ssim(denoised_img, gt_img)
        self.log("val_ssim", self.ssim)

    def test_step(self, test_batch, batch_idx):
        noisy_img, gt_img, metadata = test_batch
        batch_size, *_ = noisy_img.shape
        denoised_img = self.forward(noisy_img)

        noisy_img = F.hardtanh(noisy_img, 0.0, 1.0)
        denoised_img = F.hardtanh(denoised_img, 0.0, 1.0)

        if len(gt_img) != 0:
            gt_img = F.hardtanh(gt_img, 0.0, 1.0)
            loss = self.loss_criterion(denoised_img, gt_img)
            self.log("test_loss", loss)
            self.psnr(denoised_img, gt_img)
            self.log("test_psnr", self.psnr)
            self.ssim(denoised_img, gt_img)
            self.log("test_ssim", self.ssim)

        run = self.trainer.logger.experiment
        logger_root_path = run.repo.root_path
        experiment_name = run.experiment
        save_path = Path(logger_root_path) / experiment_name
        save_path.mkdir(parents=False, exist_ok=True)

        # Warning: this assumes that the processing steps are the same on all testing datasets, as well as the canonical name format
        test_dataset = self.trainer.datamodule.test_dataloader().dataset.datasets[0]
        process_image = test_dataset.process_image
        canonical_name = test_dataset.canonical_name
        for idx in range(batch_size):
            item_metadata = {key: value[idx] for key, value in metadata.items()}
            write_name = f"{canonical_name(**item_metadata)}"

            denoised_save = process_image(denoised_img[idx], **item_metadata)
            Image.fromarray(denoised_save).save(
                save_path / f"denoised-{write_name}.png", format="png"
            )

            denoised_nonprocessed_save = (
                denoised_img[idx].permute(1, 2, 0).detach().cpu().numpy()
            )
            tifffile.imwrite(
                save_path / f"denoised_nonprocessed-{write_name}.tiff",
                denoised_nonprocessed_save,
            )

            noisy_save = process_image(noisy_img[idx], **item_metadata)
            Image.fromarray(noisy_save).save(
                save_path / f"noisy-{write_name}.png", format="png"
            )

            if len(gt_img) != 0:
                gt_save = process_image(gt_img[idx], **item_metadata)
                Image.fromarray(gt_save).save(
                    save_path / f"gt-{write_name}.png", format="png"
                )

    def on_fit_start(self):
        val_dataloader = self.trainer.datamodule.val_dataloader()
        val_dataset = val_dataloader.dataset
        noisy_img, gt_img, metadata = [
            val_dataloader.collate_fn([el]) for el in val_dataset[len(val_dataset) - 1]
        ]

        metadata = {key: value[0] for key, value in metadata.items()}

        process_image = val_dataset.datasets[-1].process_image
        processed_noisy = process_image(noisy_img[0], **metadata)
        processed_gt = process_image(gt_img[0], **metadata)

        run = self.trainer.logger.experiment
        run.track(
            aim.Image(processed_noisy, caption="Degraded", quality=100),
            name="Validation image",
            epoch=0,
        )
        run.track(
            aim.Image(processed_gt, caption="GT", quality=100),
            name="Validation image",
            epoch=0,
        )

    def on_validation_epoch_end(self):
        val_dataloader = self.trainer.datamodule.val_dataloader()
        val_dataset = val_dataloader.dataset
        noisy_img, gt_img, metadata = [
            val_dataloader.collate_fn([el]) for el in val_dataset[len(val_dataset) - 1]
        ]
        noisy_img = noisy_img.to(self.device)
        gt_img = gt_img.to(self.device)
        denoised_img = self.forward(noisy_img)

        metadata = {key: value[0] for key, value in metadata.items()}

        process_image = val_dataset.datasets[-1].process_image
        processed_denoised = process_image(denoised_img[0], **metadata)

        run = self.trainer.logger.experiment
        run.track(
            aim.Image(processed_denoised, caption="Restored", quality=100),
            name="Validation image",
            epoch=self.trainer.current_epoch,
        )
