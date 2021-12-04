import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.nn import functional as F
from collections import defaultdict
from abc import abstractmethod
from sklearn import metrics
from scipy.special import softmax
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(nn.Module):

    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def compile(self, loss_fn, optimiser, metric_fn, precision=16):
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.metric_fn = metric_fn
        self.precision = precision
        self.scaler = torch.cuda.amp.GradScaler() if precision==16 else None
    
    def fit(self, train_dataloader, valid_dataloader=None, max_epoch=10, gpu=True):
        device = "cuda" if gpu else "cpu"
        self._check_gpu(gpu)
        self.to(device)
        for epoch in range(1, max_epoch+1):
            self.current_epoch = epoch
            try: 
                self._train_on_epoch(train_dataloader, gpu=gpu)
                
                if valid_dataloader is not None:
                    avg_loss, score = self._evaluate_on_epoch(valid_dataloader, gpu=gpu)
                    self.history['val_loss'].append(avg_loss)

            except KeyboardInterrupt:
                break

    def _train_on_epoch(self, train_dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.to(device)
        self.train()
        pbar = tqdm(train_dataloader, leave=False)
        losses = 0
        for step, (x, y) in enumerate(pbar):
            pbar.set_description(f'Epoch {self.current_epoch}')
            self.optimiser.zero_grad()
            x, y = x.to(device), y.to(device)

            if self.precision == 16:
                with torch.cuda.amp.autocast():
                    pred = self(x)
                    loss = self.loss_fn(pred, y)
                    losses += loss.item()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
            elif self.precision == 32:
                loss.backward()
                self.optimiser.step()
                
            torch.cuda.empty_cache()
            pbar.set_postfix({'train_loss':loss.item()})
        avg_loss = losses / len(train_dataloader.dataset)
        self.history['train_loss'].append(avg_loss)

    def _evaluate_on_epoch(self, valid_dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.to(device)
        with torch.no_grad():
            self.eval()
            losses, correct = 0, 0
            y_hats, targets = [], []
            for x, y in valid_dataloader:
                x, y = x.to(device), y.to(device)
                pred = self(x)
                loss = self.loss_fn(pred, y)
                losses += loss.item()
                y_hat = torch.max(pred, 1)[1]
                y_hats += y_hat.tolist()
                targets += y.tolist()
                correct += (y_hat == y).sum().item()
        avg_loss = losses / len(valid_dataloader.dataset)
        score = self.metric_fn(targets, y_hats)
        torch.cuda.empty_cache()
        return avg_loss, score
    
    def predict_proba(self, test_dataloader, gpu=True):
        device = "cuda" if gpu else "cpu"
        self.eval()
        self.to(device)
        y_probs = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader, leave=False), 0):
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = self(inputs)
                y_probs.extend(outputs.detach().cpu())
        y_probs = np.vstack(y_probs)
        logits = softmax(y_probs, axis=1)
        return logits
    
    def predict(self, test_dataloader, gpu=True):
        y_prob = self.predict_proba(test_dataloader, gpu=gpu)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred
    
    def evaluate(self, test_dataloader, gpu=True):
        avg_loss, score = self._evaluate_on_epoch(test_dataloader, gpu=gpu)
        return avg_loss, score

    def _check_gpu(self, gpu):
        gpu_available = 'True' if torch.cuda.is_available() else 'False'
        gpu_used = 'True' if gpu else 'False'
        logger.info(f'GPU available: {gpu_available}, used: {gpu_used}')

    def plot(self):
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.history['train_loss'])), self.history['train_loss'], label="train")
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'], label="valid")
        plt.title("Loss (per epoch)")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()
