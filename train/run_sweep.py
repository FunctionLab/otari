import argparse

from pytorch_lightning import Trainer, seed_everything, LightningDataModule
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import random_split
import torch
import wandb
import yaml
from torch_geometric.data import DataLoader

from abundance_model import IsoAbundanceGNN
from data import IsoAbundanceDataset


class IsoAbundanceDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, configs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = configs['batch_size']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    

def train():
    wandb.init()
    logger = WandbLogger(name='abundance_sweep', project='abundance_sweep', dir='/mnt/home/alitman/ceph/isoModel_w_gnn')
    configs = wandb.config

    # load data
    dataset_path = configs['dataset_loc']
    dataset = IsoAbundanceDataset(root=dataset_path, file_name='all_transcript_graphs_directed_espresso.pt')
    test_dataset = [d for d in dataset if d.chrom in [8]]

    # split rest of data into training and validation
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[configs['train_size'], configs['val_size']],
        generator=torch.Generator()
    )

    data_module = IsoAbundanceDataModule(train_dataset, val_dataset, test_dataset, configs)

    # create model
    model = IsoAbundanceGNN(
        dataset.num_features, 
        configs['hidden_channels'], 
        configs['num_tissues'], 
        lr=configs['lr'],
        num_heads=configs['num_heads'],
        max_epochs=configs['max_epochs'],
        batch_size=configs['batch_size'],
        dp_rate=configs['dp_rate'],
        tissue_names=configs['tissue_names']
        )

    # start training
    trainer = Trainer(default_root_dir="../ceph/isoModel_w_gnn/model_checkpoints/",
                      max_epochs=configs['max_epochs'],
                      accelerator="gpu", devices="auto",
                      enable_checkpointing=True,
                      log_every_n_steps=128, logger=logger,
                      strategy=DDPStrategy(find_unused_parameters=True)
                      )
    trainer.fit(model, data_module)
    trainer.test(model=model, dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config_path', 
        type=str,
        help='path to yaml config file', 
        required=True
    )
    args = parser.parse_args()
    with open(args.config_path, 'r') as file:
        configs = yaml.safe_load(file)

    seed_everything(configs['parameters']['seed']['value'], workers=True)

    sweep_id=wandb.sweep(configs, project="abundance_sweep")
    wandb.agent(sweep_id=sweep_id, function=train, count=12)
