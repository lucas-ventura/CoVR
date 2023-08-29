import shutil

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.tools.files import json_dump


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()

    if fabric.global_rank == 0:
        json_dump(OmegaConf.to_container(cfg, resolve=True), "hydra.json")

    model = instantiate(cfg.model)
    model = fabric.setup(model)

    for dataset in cfg.test:
        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Testing {cfg.test[dataset].dataname}".center(columns))

        data = instantiate(cfg.test[dataset])
        test_loader = fabric.setup_dataloaders(data.test_dataloader())

        test = instantiate(cfg.test[dataset].test)
        test(model, test_loader, fabric=fabric)


if __name__ == "__main__":
    main()
