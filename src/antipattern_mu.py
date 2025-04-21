"""
    Contains the base pipeline of the Anti-Pattern MU Algorithms
"""

# import required libraries
import os

from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from logger import TimeStampLogger, LossAccuracyLogger

from abc import abstractmethod, ABC
import random
from tqdm import tqdm

torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MixedDataset(Dataset):
    def __init__(self, datasets_forget, datasets_retain, ratio: float = 1.0):
        self.forget = datasets_forget
        self.retain = datasets_retain
        self.ratio = int(len(datasets_forget) * ratio + 0.5)

    def __getitem__(self, i):

        threshold = len(self.forget)
        if i < threshold:
            return self.forget[i]
        else:
            return self.retain[random.randint(0, len(self.retain) - 1)]

    def __len__(self):
        return len(self.forget) + self.ratio

class AntiPatternMU(ABC):
    
    def __init__(self,
            model: nn.Module,
            data_forget: Dataset,
            data_retain: Dataset,
            data_test: Dataset,
            logs: bool = True,
            hyperparameters: Dict = {},
        ):

        self.model = model
        self.data_forget = data_forget
        self.data_retain = data_retain
        self.data_test = data_test
        self.logs = logs
        self.hyperparameters = hyperparameters

        if logs:
            self.timer = TimeStampLogger()
            self.loss_accs_logger = LossAccuracyLogger(dataset = self.data_test)

    @abstractmethod
    def noise_maximizer(self) -> Dataset:
        pass

    def impairing_phase(self, antipatterns_ds):
        noisy_loader = DataLoader(
            dataset=MixedDataset(
                datasets_forget = antipatterns_ds,
                datasets_retain = self.data_retain,
                ratio=self.hyperparameters['impairment_ratio'] if 'impairment_ratio' in self.hyperparameters else 1.0
                ),
            batch_size=os.getenv("BATCHSIZE"), 
            shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hyperparameters['lr_impairing'] if 'lr_impairing' in self.hyperparameters else 0.02) # Hyperparameter

        for epoch in range(1):
            self.model.train(True)
           
            for i, data in enumerate(noisy_loader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

    def repairing_phase(self,):

        heal_loader = torch.utils.data.DataLoader(self.data_retain, batch_size=os.getenv("BATCHSIZE"), shuffle = True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hyperparameters['lr_repairing'] if 'lr_repairing' in self.hyperparameters else 0.01) # Hyperparameter

        for epoch in range(1):
            self.model.train(True)
                        
            for i, data in enumerate(heal_loader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

    def run(self):
    
        if self.logs:
            self.loss_accs_logger.log(self.model, "Initial Scores:")
            self.timer.start()
        antipatters_ds = self.noise_maximizer()
        if self.logs:
            self.timer.start()
        self.impairing_phase()
        if self.logs:
            self.timer.logs("Impairment Time")
            self.loss_accs_logger.log(self.model, "Post Impairment Scores:")
        self.repairing_phase()
        if self.logs:
            self.timer.logs("Repairment Time")
            self.timer.end()
            self.loss_accs_logger.log(self.model, "Post Repairment Scores:")
    
        return self.model
