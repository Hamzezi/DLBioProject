import hydra
from omegaconf import OmegaConf

from run import *

@hydra.main(version_base=None, config_path='conf', config_name='main')
def model_params(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    _, _, model = initialize_dataset_model(cfg)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model_params()