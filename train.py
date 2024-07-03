from lightning.pytorch.cli import LightningCLI
import warnings
import torch.multiprocessing
import torch

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')

def main():

    cli = LightningCLI(
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )
    
if __name__ == "__main__":

    main()