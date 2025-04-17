from collections import defaultdict

import torch
import numpy as np
import wandb
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, auc
import torch.nn.functional as F
from scipy.stats import spearmanr
from pytorch_metric_learning import miners

from utils.utils import binarize, compute_tissue_cutoffs


class GNN_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dp_rate=0.3):
        """
        Initializes the GNN_Block model, a Graph Neural Network (GNN) block designed for isoform abundance prediction.

        This block uses two layers of GATv2Conv (Graph Attention Network v2) for message passing, 
        with batch normalization applied after each convolutional layer. A dropout layer is included 
        to prevent overfitting before the final linear layer.

            in_channels (int): The dimensionality of the input node features.
            hidden_channels (int): The dimensionality of the hidden node features.
            out_channels (int): The dimensionality of the output node features (e.g., number of classes).
            num_heads (int): The number of attention heads for the first GATv2Conv layer.
            dp_rate (float, optional): The dropout rate applied before the final linear layer. Defaults to 0.3.
        """

        super(GNN_Block, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class Otari(nn.Module):
    def __init__(self, dp_rate_linear=0.5):
        super(Otari, self).__init__()
        
        self.IsoGNNModel1 = GNN_Block(23600, 512, 512, 2, dp_rate=0.5)
        self.IsoGNNModel2 = GNN_Block(512, 512 // 2, 512 // 2, 2, dp_rate=0.5)
        self.IsoGNNModel3 = GNN_Block(512 // 2, 512 // 4, 512 // 4, 2, dp_rate=0.5)
        
        self.linear1 = nn.Linear(23600, 512)
        self.linear2 = nn.Linear(512, 512 // 2)
        self.linear3 = nn.Linear(512 // 2, 512 // 4)
        
        self.layernorm1 = nn.LayerNorm(512)
        self.layernorm2 = nn.LayerNorm(512 // 2)
        self.layernorm3 = nn.LayerNorm(512 // 4)

        self.dropout = nn.Dropout(0.3)
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(512 // 4, 512 // 4),
            nn.ReLU(),
            nn.Dropout(dp_rate_linear),
            nn.Linear(512 // 4, 30)
        )

    def forward(self, batch):
        out1 = self.IsoGNNModel1(batch.x, batch.edge_index)
        lout1 = self.layernorm1(out1)
        
        out2 = self.IsoGNNModel2(lout1, batch.edge_index)
        rout2 = out2 + self.linear2(lout1)  # Residual connection
        lout2 = self.layernorm2(rout2)

        out3 = self.IsoGNNModel3(lout2, batch.edge_index)
        rout3 = out3 + self.linear3(lout2)  # Residual connection
        lout3 = self.layernorm3(rout3)
        
        # Global pooling and output head
        outpool = global_max_pool(lout3, batch.batch_idx)
        outpool = self.dropout(outpool)
        predict = self.head(outpool)
        
        return predict


class IsoAbundanceGNN(pl.LightningModule):    
    def __init__(self, in_channels, hidden_channels, num_tissues, lr, num_heads, max_epochs, batch_size, dp_rate, tissue_names):
        """
        Initializes the IsoAbundanceModel class.
        Args:
            in_channels (int): Number of input channels for the model.
            hidden_channels (int): Number of hidden channels in the model.
            num_tissues (int): Number of tissue types to predict.
            lr (float): Learning rate for the optimizer.
            num_heads (int): Number of attention heads in the model.
            max_epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
            dp_rate (float): Dropout rate for the model.
            tissue_names (list of str): List of tissue names for prediction.
        Attributes:
            model (IsoAbundanceModel): The main model architecture.
            criterion (nn.MSELoss): Mean squared error loss function.
            lr (float): Learning rate for the optimizer.
            max_epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
            num_tissues (int): Number of tissue types to predict.
            tissues (list of str): List of tissue names for prediction.
            miner (miners.BatchHardMiner): Miner for hard example mining.
            triplet_loss (nn.TripletMarginLoss): Triplet margin loss for mining examples.
            test_y_true (defaultdict): Stores true values for the test set for each tissue.
            test_y_scores (defaultdict): Stores predicted values for the test set for each tissue.
            tissues_to_values (defaultdict): Stores tissue-specific values.
            cutoffs (dict): ESPRESSO cutoffs for tissue-specific thresholds.
            wd (float): Weight decay for the optimizer.
        Notes:
            - Initializes a Weights and Biases (wandb) project for logging.
            - Logs the model architecture to wandb.
            - Computes tissue-specific cutoffs using the `compute_tissue_cutoffs` function.
        """

        super().__init__()
        self.save_hyperparameters()

        self.model = Otari()
        
        wandb.init(project='otari', name='otari', dir='./otari')
        model_architecture = str(self.model)

        self.criterion = nn.MSELoss()
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_tissues = num_tissues
        self.tissues = tissue_names

        # Hard miner and loss for mining examples
        self.miner = miners.BatchHardMiner()
        self.triplet_loss = nn.TripletMarginLoss()
        
        self.test_y_true = defaultdict(list)
        self.test_y_scores = defaultdict(list)
        self.tissues_to_values = defaultdict(list)

        wandb.log({"model_architecture": model_architecture})
        
        self.cutoffs = compute_tissue_cutoffs()
        self.wd = 5e-4

    def forward(self, data):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        data.y = data.y.view(-1, self.num_tissues)
        loss = self.criterion(x, data.y) 
        mae = torch.mean(torch.abs(x - data.y))
        spearman_per_tissue = [spearmanr(data.y[:, i].cpu().detach().numpy(), x[:, i].cpu().detach().numpy())[0] for i in range(self.num_tissues)]

        return loss, x, data.y, mae, spearman_per_tissue
        
    def configure_optimizers(self):
        wandb.log({'wd': self.wd})
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            ),
            "interval": "epoch",
            "frequency": 2,
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler,
        } 

        
    def log_metrics(self, stage, x, y, tissue, batch_size):
        self.log(f'{stage}/rocauc_{tissue}', roc_auc_score(y, x), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        y_pred = [1 if score > 0.5 else 0 for score in x]
        self.log(f'{stage}/precision_{tissue}', precision_score(y, y_pred), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}/recall_{tissue}', recall_score(y, y_pred), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        loss, x, y, mae, spearman_per_tissue = self.forward(batch, mode='train')
        sq_error = (x - batch.y) ** 2
        mse = sq_error.mean(dim=1)
        weighted_error = mse / torch.max(mse)
        error_labels = torch.where(weighted_error > 0.5, 1, 0)
        anchors, pos, neg = self.miner(x, error_labels)
        hard_loss = self.triplet_loss(x[anchors], x[pos], x[neg])
        
        self.log('train/loss', loss, batch_size=batch.x.size(0))
        self.log('train/mae', mae, batch_size=batch.x.size(0))
        self.log('train/spearmanr', np.mean(spearman_per_tissue), batch_size=batch.x.size(0))
        self.log('train/hard_loss', hard_loss, batch_size=batch.x.size(0))
        self.log('train/total_loss', loss + 0.25 * hard_loss, batch_size=batch.x.size(0))
        
        return loss + (0.25 * hard_loss)
    
    def validation_step(self, batch, batch_idx):
        loss, x, y, mae, spearman_per_tissue = self.forward(batch, mode='val')
        sq_error = (x - batch.y) ** 2
        mse = sq_error.mean(dim=1)
        weighted_error = mse / torch.max(mse)
        error_labels = torch.where(weighted_error > 0.5, 1, 0)
        anchors, pos, neg = self.miner(x, error_labels)
        hard_loss = self.triplet_loss(x[anchors], x[pos], x[neg])
        
        self.log('val/mae', mae, batch_size=batch.x.size(0))
        self.log('val/loss', loss, batch_size=batch.x.size(0))
        self.log('val/spearmanr', np.mean(spearman_per_tissue), batch_size=batch.x.size(0))
        self.log('val/hard_loss', hard_loss, batch_size=batch.x.size(0))
        self.log('val/total_loss', loss + 0.25 * hard_loss, batch_size=batch.x.size(0))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, x, y, _, spearman_per_tissue = self.forward(batch, mode='test')
        y_copy = y.cpu().detach().numpy()
        x_copy = x.cpu().detach().numpy().reshape(-1, self.num_tissues)

        # convert to binarized array with cutoffs
        for i, tissue in zip(range(len(self.tissues)), self.tissues):
            self.log(f'test/spearmanr_{tissue}', spearman_per_tissue[i], batch_size=batch.x.size(0))
            
            y_tissue = y_copy[:, i]
            y_tissue = np.array([binarize(j, tissue, self.cutoffs) for j in y_tissue])
            nan_pos = np.isnan(y_tissue)
            y_tissue = y_tissue[~nan_pos]

            x_tissue_copy = x_copy[:, i]
            x_tissue = x_tissue_copy[~nan_pos]

            self.test_y_true[tissue].extend(y_tissue)
            self.test_y_scores[tissue].extend(x_tissue)
            self.tissues_to_values[tissue].extend(x_tissue_copy) 
              
        return loss

    def make_prediction(self, graph):
        pl.seed_everything(42, workers=True)
        x, edge_index, batch_idx = graph.x, graph.edge_index, graph.batch
        x = self.model(x, edge_index, batch_idx)

        return x, graph.y
            

class IsoAbundanceDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, configs):
        """
        Loads training, validation, and test datasets, as well as configuration settings.
        Args:
            train_dataset (Dataset): The dataset used for training the model.
            val_dataset (Dataset): The dataset used for validating the model during training.
            test_dataset (Dataset): The dataset used for testing the model after training.
            configs (Config): A configuration object containing training parameters, including batch size.
        Attributes:
            train_dataset (Dataset): Stores the training dataset.
            val_dataset (Dataset): Stores the validation dataset.
            test_dataset (Dataset): Stores the test dataset.
            batch_size (int): The batch size used for training, extracted from the configuration.
        """

        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = configs.train.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    

class PrecisionRecallPlotCallback(pl.Callback):
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
    
    def on_test_epoch_end(self, trainer, pl_module):
        all_fprs = []
        all_tprs = []
        aucs = []
        tissues = list(pl_module.test_y_true.keys())
        
        for _, tissue in zip(range(len(tissues)), tissues):
            y_true = pl_module.test_y_true[tissue]
            y_scores = pl_module.test_y_scores[tissue]
            tissue_to_values = pl_module.tissues_to_values[tissue]
            tissue_to_values_sorted = sorted(tissue_to_values)
            
            y_scores_ranked = np.array([(tissue_to_values_sorted.index(j))/len(tissue_to_values_sorted) for j in y_scores])
            pl_module.log_metrics('test', y_scores_ranked, y_true, tissue, pl_module.batch_size)

            fpr, tpr, _ = roc_curve(y_true, y_scores_ranked)
            roc_auc = auc(fpr, tpr)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            aucs.append(roc_auc)

            y_score_transformed = [[1 - score, score] for score in y_scores_ranked]
            wandb.log({f"test/{tissue}_pr_curve": wandb.plot.pr_curve(y_true, y_score_transformed, labels=None, classes_to_plot=None, title=f"{tissue} Precision v. Recall")})
            wandb.log({f"test/{tissue}_roc_curve": wandb.plot.roc_curve(y_true, y_score_transformed, labels=None, classes_to_plot=None, title=f"{tissue} ROC")})

        # Reset the lists after logging
        for tissue in tissues:
            pl_module.test_y_true[tissue] = []
            pl_module.test_y_scores[tissue] = []
            pl_module.tissues_to_values[tissue] = []
