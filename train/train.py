import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
import torch

from model.otari import IsoAbundanceGNN, IsoAbundanceDataModule, PrecisionRecallPlotCallback
from utils.data import IsoAbundanceDataset
from utils.utils import load_config


def train(configs):
    logger = WandbLogger(name='otari', project='otari', dir='./logs')
    
    # load data
    dataset_path = configs.dataset.loc
    dataset = IsoAbundanceDataset(root=dataset_path, file_name='all_transcript_graphs_directed_espresso.pt')

    # hold out chromosome 8 for testing
    rest_of_dataset = [d for d in dataset if d.chrom not in [8]]
    test_dataset = [d for d in dataset if d.chrom in [8]]

    print("Number of samples in the dataset: ", len(dataset))
    
    # split rest of data into train and validation sets
    train_dataset, val_dataset = random_split(
        dataset=rest_of_dataset,
        lengths=[configs.dataset.train_prop, configs.dataset.val_prop],
        generator=torch.Generator()
    )

    data_module = IsoAbundanceDataModule(train_dataset, val_dataset, test_dataset, configs)

    # create model
    model = IsoAbundanceGNN(
        dataset.num_features, 
        configs.model.args.hidden_channels, 
        len(configs.model.args.tissue_names), 
        lr=configs.model.args.lr,
        num_heads=configs.model.args.num_heads,
        max_epochs=configs.train.max_epochs,
        batch_size=configs.train.batch_size,
        dp_rate=configs.model.args.dp_rate,
        tissue_names=configs.model.args.tissue_names
    )

    checkpoint_callback = ModelCheckpoint(dirpath="./model_checkpoints", save_top_k=1, monitor="val/mae", mode="min")

    # start training
    trainer = Trainer(default_root_dir="./model_checkpoints",
                      max_epochs=configs.train.max_epochs,
                      accelerator="gpu", devices="auto",
                      enable_checkpointing=True,
                      log_every_n_steps=configs.train.log_every_n_steps, logger=logger,
                      callbacks=[checkpoint_callback, PrecisionRecallPlotCallback(logger)],
                      strategy=DDPStrategy(find_unused_parameters=True)
                      )
    trainer.fit(model, data_module)
    trainer.test(ckpt_path="best", model=model, dataloaders=data_module.test_dataloader())

    # save model weights
    if configs.save_model:
        model_path = configs.model_save_path
        torch.save(model, model_path)


def main(yaml_path):
    configs = load_config(yaml_path)
    seed_everything(configs.seed, workers=True)
    train(configs)


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
    main(args.config_path)
