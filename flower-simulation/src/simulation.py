import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from test_transformers.client_app import create_client_app
from test_transformers.server_app import create_server_app
import torch
from flwr.simulation import run_simulation


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
log = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    backend_config = cfg.get("backend_config")

    if DEVICE.type == "cuda":
        backend_config = cfg.get("gpu_backend_config")

    NUM_PARTITIONS = cfg.get("partitions")

    # Run simulation
    run_simulation(
        server_app=create_server_app(cfg),
        client_app=create_client_app(cfg),
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )

if __name__ == "__main__":
    my_app()
