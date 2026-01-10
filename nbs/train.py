from antipasto.train.train_adapter import train_model
from antipasto.config import proj_root, TrainingConfig, default_configs
import tyro 

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs, use_underscores=True)
    train_model(config)
