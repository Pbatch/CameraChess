import os
from glob import glob

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data
import torchmetrics
import wandb
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from camera_chess.constants import ROOT_DIR, CLASSES


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mode):
        self.image_paths = image_paths

        base_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        if mode == 'train':
            self.transform = A.Compose([
                A.Flip(),
                base_transform
            ])
        elif mode == 'val':
            self.transform = base_transform
        else:
            raise ValueError(f'Mode {mode} is not supported')

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = np.array(Image.open(image_path))
        image = self.transform(image=image)['image']
        image.requires_grad = True

        label = CLASSES.index(os.path.basename(image_path).split('category=')[1].split('_')[0])

        return image, label

    def __len__(self):
        return len(self.image_paths)


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.all_image_paths = sorted(glob(os.path.join(ROOT_DIR, 'data', 'classification', '*.jpg')))

    def train_dataloader(self):
        image_paths = [p for p in self.all_image_paths
                       if os.path.basename(p).startswith('roboflow')]
        train_dataset = Dataset(image_paths, 'train')
        return DataLoader(train_dataset,
                          batch_size=8,
                          shuffle=True)

    def val_dataloader(self):
        image_paths = [p for p in self.all_image_paths
                       if not os.path.basename(p).startswith('roboflow')]
        val_dataset = Dataset(image_paths, 'val')
        return DataLoader(val_dataset,
                          batch_size=8,
                          shuffle=False)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.num_classes = len(CLASSES)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.model = None

    def setup(self, stage=None):
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT,
                                        progress=True)
        self.model.classifier[-1] = nn.Linear(1280, len(CLASSES))
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith('classifier.3')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)

        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)

        loss = self.loss(logits, y)
        self.log("val_loss", loss)

        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

        return {'y_true': y, 'preds': logits.argmax(dim=1)}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()

        conf_mat = wandb.plot.confusion_matrix(y_true=y_true,
                                               preds=preds,
                                               class_names=CLASSES)
        wandb.log({"conf_mat": conf_mat})


def main():
    torch.set_float32_matmul_precision('medium')
    model = Model()
    datamodule = DataModule()
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=10,
                         accelerator='gpu',
                         devices=1,
                         precision=16,
                         logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
