"""
    This file shall handle all the logging for the project
"""

import datetime
import os
from torhch.nn import Module
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod

from metrics import evaluate_loss_accs

class TimeStampLogger:
    def __init__(self):
        self.start_time = None

    def start(self):
        """Starts the timer"""
        self.start_time = datetime.datetime.now().timestamp()

    def stop(self):
        """Stops the timer temporarily"""
        pass # TODO

    def end(self):
        """Stops the timer"""
        self.start_time = None

    def log(self, message:str):
        """Logs the time, with an optional message"""
        if self.start_time is None:
            return
        print(message)
        print("Time: {}".format(datetime.datetime.now().timestamp() - self.start_time))

class Logger(ABC):
    @abstractmethod
    def log(self,):
        pass

class LossAccuracyLogger(Logger):

    def __init__(self, dataset: Dataset):
        self.dataloader = DataLoader(dataset, batch_size=os.getenv("BATCHSIZE"), shuffle = True)
        self.accs = {}
        self.losses = {}

    def log(self, model: Module, message: str):
        
        scores = evaluate_loss_accs(model, self.dataloader,)

        print(message)
        print("Accuracy: {}".format(scores["Acc"]*100))
        print("Loss: {}".format(scores[0]["Loss"]))
        self.losses[len(self.losses)] = scores["Losses"]
        self.accs[len(self.accs)] = scores["Acc"]

    def get_losses(self):
        return self.losses
    
    def get_accs(self):
        return self.accs