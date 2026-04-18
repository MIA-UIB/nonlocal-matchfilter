import hydra
import pyrootutils
import torch
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(version_base="1.3", config_path=str(root / "conf"), config_name="train")
def main(cfg: DictConfig):

    if cfg.resume_training and cfg.ckpt_path is None:
        raise HydraException("ckpt_path must be provided when resuming training")

    if cfg.resume_training:
        data_structure_class_name = OmegaConf.select(cfg, "data._target_")
        OmegaConf.update(
            cfg, "data._target_", f"{data_structure_class_name}.load_from_checkpoint"
        )
        data_module = hydra.utils.instantiate(cfg.data, checkpoint_path=cfg.ckpt_path)
        model_structure_class_name = OmegaConf.select(cfg, "model._target_")
        OmegaConf.update(
            cfg, "model._target_", f"{model_structure_class_name}.load_from_checkpoint"
        )
        model = hydra.utils.instantiate(cfg.model, checkpoint_path=cfg.ckpt_path)
    else:
        data_module = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model)
        if cfg.ckpt_path is not None:
            model.load_state_dict(
                torch.load(cfg.ckpt_path, weights_only=False)["state_dict"], strict=True
            )

    callbacks = [
        hydra.utils.instantiate(callback_config)
        for _, callback_config in cfg.callbacks.items()
    ]
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=cfg.ckpt_path if cfg.resume_training else None,
    )


if __name__ == "__main__":
    main()
