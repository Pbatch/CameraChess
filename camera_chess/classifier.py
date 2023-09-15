import os
from glob import glob

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

from camera_chess.constants import CLASSES, CLASSIFIER_DIR


def load_model(checkpoint_path):
    # model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                k = k.replace('model.', '', 1)
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    return model


class ChessDataset(torch.utils.data.Dataset):
    # transform = MobileNet_V3_Large_Weights.DEFAULT.transforms()
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    def __init__(self, split):
        self.split = split
        self.paths = glob(os.path.join(CLASSIFIER_DIR, split, '*.jpg'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image_path = self.paths[item]

        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)

        basename = os.path.basename(image_path)
        cls = os.path.splitext(basename)[0].split('_')[-1]
        y = CLASSES.index(cls)

        return x, y


class DataModule(pl.LightningModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        self.dataloader_kwargs = {'batch_size': self.batch_size,
                                  'num_workers': 4}

    def setup(self, stage):
        return None

    def train_dataloader(self):
        dataset = ChessDataset('train')
        dataloader = DataLoader(dataset,
                                shuffle=True,
                                **self.dataloader_kwargs)
        return dataloader

    def val_dataloader(self):
        dataset = ChessDataset('val')
        dataloader = DataLoader(dataset,
                                shuffle=False,
                                **self.dataloader_kwargs)
        return dataloader


class MobileNet(pl.LightningModule):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path

        self.model = load_model(checkpoint_path)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("val_acc", self.valid_acc,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc,
                 on_epoch=True,
                 on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


def main():
    torch.set_float32_matmul_precision('medium')
    model = MobileNet()
    datamodule = DataModule()
    logger = WandbLogger(project="ChessClassifier")

    early_stopping_callback = EarlyStopping('val_loss',
                                            patience=5)
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="val_loss")
    trainer = pl.Trainer(precision='16-mixed',
                         logger=logger,
                         callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model=model,
                datamodule=datamodule)


if __name__ == '__main__':
    main()