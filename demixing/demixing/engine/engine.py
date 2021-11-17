import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from tqdm.auto import tqdm
from torch.nn import functional as F
from collections import OrderedDict
from scipy.special import softmax


class ClassificationEngine(pl.LightningModule):

    def __init__(
            self, 
            backbone, 
            learning_rate=1e-4, 
            weight_decay=0.0005
        ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

        # Save the logs to visualise
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).float()
        return {'loss': loss, 'acc': acc, 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.train_losses.append(avg_loss.detach().cpu().item())
        self.train_accuracies.append(train_acc.detach().cpu().item())
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        acc = (logits.argmax(-1) == y).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        out = {'val_loss': avg_loss, 'val_acc': val_acc}
        self.valid_losses.append(avg_loss.detach().cpu().item())
        self.valid_accuracies.append(val_acc.detach().cpu().item())
        torch.cuda.empty_cache()
        return {**out, 'log': out}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            betas=(0.95, 0.999), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

    def predict_proba(self, test_dl):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        self.to(device)
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dl), 0):
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = self(inputs)
                y_probs.extend(outputs.detach().cpu().numpy())
        return softmax(np.vstack(y_probs), axis=1)

    def predict(self, test_dl):
        y_prob = self.predict_proba(test_dl)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def plot(self):
        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.train_losses)), self.train_losses, label="train")
        plt.plot(range(len(self.valid_losses)), self.valid_losses, label="valid")
        plt.title("Loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.train_accuracies)), self.train_accuracies, label="train")
        plt.plot(range(len(self.valid_accuracies)), self.valid_accuracies, label="valid")
        plt.title("Accuracy")
        plt.legend(loc="upper right")
        plt.grid()
        plt.tight_layout()
        plt.show()