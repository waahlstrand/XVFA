from lightning.pytorch.cli import LightningCLI
from data.superb import SuperbDataModule
import torch.multiprocessing
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('medium')

def main():

    cli = LightningCLI(
        # SpineDINO,
        datamodule_class=SuperbDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )

    
if __name__ == "__main__":

    main()