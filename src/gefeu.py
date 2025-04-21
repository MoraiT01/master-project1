"""
    Generator Feature Unlearning
    Handle the creation process of the anti-pattern, extending 'antipattern_mu.py'
"""
# import required libraries
from dotenv import load_dotenv
import numpy as np
load_dotenv()

from typing import Literal, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import math
from antipattern_mu import AntiPatternMU

torch.manual_seed(100)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class NoiseGenerator(nn.Module):
    """
    A neural network module for generating noise patterns
    through a series of fully connected layers.
    """

    def __init__(
            self, 
            dim_out: list,
            dim_hidden: list = [1000],
            dim_start: int = 100,
            ):
        """
        Initialize the NoiseGenerator.

        Parameters:
        dim_out (list): The output dimensions for the generated noise.
        dim_hidden (list): The dimensions of hidden layers, defaults to [1000].
        dim_start (int): The initial dimension of random noise, defaults to 100.
        """
        super().__init__()
        self.dim = dim_out
        self.start_dims = dim_start  # Initial dimension of random noise
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Define fully connected layers
        self.layers = {}
        self.layers["l1"] = nn.Linear(self.start_dims, dim_hidden[0]).to(self.device)
        last = dim_hidden[0]
        for idx in range(len(dim_hidden)-1):
            self.layers[f"l{idx+2}"] = nn.Linear(dim_hidden[idx], dim_hidden[idx+1]).to(self.device)
            last = dim_hidden[idx+1]

        # Define output layer
        self.f_out = nn.Linear(last, math.prod(self.dim)).to(self.device)        

    def forward(self):
        """
        Forward pass to transform random noise into structured output.

        Returns:
        torch.Tensor: The reshaped tensor with specified output dimensions.
        """
        # Generate random starting noise
        x = torch.randn(self.start_dims).to(self.device)
        x = x.flatten()

        # Transform noise into learnable patterns
        for layer in self.layers.keys():
            x = self.layers[layer](x)
            x = torch.relu(x)

        # Apply output layer
        x = self.f_out(x)

        # Reshape tensor to the specified dimensions
        reshaped_tensor = x.view(self.dim)
        return reshaped_tensor

class NoiseDataset(Dataset):
    """
    A DataLoader which uses a noise generator to generate data and labels.

    Args:
        noise_generator (NoiseGenerator): The noise generator to use.
        noise_labels (Dict[int, torch.Tensor] | torch.Tensor): The labels to use. If a tensor, it is used as the labels for all samples. If a dict, it is used as a mapping of indices to labels.
        number_of_noise (int, optional): The number of noise samples to generate. Defaults to 100.
    """

    def __init__(self, noise_generator: NoiseGenerator, noise_labels: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int = 100,):

        self.noise_generator = noise_generator
        self.noise_labels  = noise_labels
        self.number_of_noise = number_of_noise if isinstance(self.noise_labels, torch.Tensor) else len(self.noise_labels)

    def __len__(self) -> int:
        """
        The number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.number_of_noise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and the label.
        """
        return self.noise_generator().to(DEVICE), self.noise_labels.to(DEVICE) if isinstance(self.noise_labels, torch.Tensor) else self.noise_labels[idx]
        

class GeFeU(AntiPatternMU):

    def __init__(self,
        model: nn.Module,
        data_forget: Dataset,
        data_retain: Dataset,
        data_test: Dataset,
        logs: bool = True,
        hyperparameters: Dict = {},
            ):
        super.__init__(model, data_forget, data_retain, data_test, logs, hyperparameters)

    def prep_noise_generator(self) -> Tuple[NoiseGenerator, Dict, Dict]:
        """
        Prepares a NoiseGenerator and two dictionaries containing the original labels and the created labels.

        Args:
            forget_data (Dataset): The dataset containing the samples to be forgotten.
            model (torch.nn.Module): The model to be used for generating the created labels.
            t_Layers (list): The dimensions of the hidden layers in the NoiseGenerator.
            t_Noise_Dim (int): The number of features in the generated noise.

        Returns:
            Tuple[NoiseGenerator, Dict, Dict]: A tuple containing the NoiseGenerator, the original labels and the created labels.
        """
        
        # Create two dictionaries to store the original labels and the created labels
        noises = {}
        og_labels = {}
        created_labels =  {}

        # Set the model to evaluation mode
        self.model.eval()

        # Iterate over the forget_data and create the created labels
        for index, data in enumerate(DataLoader(self.data_forget, batch_size=1, shuffle=False)):
            # Get the sample and the label
            s, l = data

            # Use the model to generate the created label
            new_l = F.softmax(self.model(s.to(DEVICE)).detach(), dim=1)
            
            # Store the created label and the original label
            created_labels[index] = new_l[0].to(DEVICE)
            og_labels[index] = l[0].to(DEVICE)

        # Create a NoiseGenerator with the specified dimensions
        noises = NoiseGenerator(
                s[0].shape,
                dim_hidden=self.hyperparameters["t_Layers"] if "t_Layers" in self.hyperparameters else [1000],
                dim_start=self.hyperparameters["t_Noise_Dim"] if "t_Noise_Dim" in self.hyperparameters else 100,
            ).to(DEVICE)

        # Return the NoiseGenerator and the two dictionaries
        return noises, created_labels, og_labels
    
    def noise_maximization(self) -> Tuple[NoiseGenerator, Dict[int, torch.Tensor] | torch.Tensor]:
        """
        This function maximizes the loss of the model on the forget_data by generating noise with the noise generator
        and adding it to the dataset.

        Args:
            forget_data (Dataset): The dataset to use for the unlearning process.
            model (torch.nn.Module): The model to be unlearned.
            t_Epochs (int): The number of epochs to train the noise generator.
            t_Learning_Rate (float): The learning rate of the noise generator.
            t_Batch_Size (int): The batch size of the DataLoader.
            t_Regularization_term (float): The regularization term to add to the loss.
            t_Layers (list): The layers of the noise generator.
            t_Noise_Dim (int): The dimension of the noise to generate.
            logs (bool, optional): Whether to print logs. Defaults to False.

        Returns:
            Tuple[NoiseGenerator, Dict[int, torch.Tensor] | torch.Tensor]: A tuple containing the noise generator and the labels of the forget_data.
        """
        noise_generator, created_labels, og_labels = self.prep_noise_generator()
        noise_loader = DataLoader(
            dataset=NoiseDataset(noise_generator, og_labels),
            batch_size=self.hyperparameters["t_Batch_Size"] if "t_Batch_Size" in self.hyperparameters else 100, # Hyperparameter
            shuffle=True,
        )

        self.model.to(DEVICE)
        self.model.eval()
        optimizers = torch.optim.Adam(noise_generator.parameters(), lr = self.hyperparameters["t_Learning_Rate"] if "t_Learning_Rate" in self.hyperparameters else 0.03) # Hyperparameter
        num_epochs = self.hyperparameters["t_Epochs"] if "t_Epochs" in self.hyperparameters else 5 # Hyperparameter

        epoch = 0
        while True:
            total_loss = []
            epoch += 1
            for input_batch, l in noise_loader:

                outputs = self.model(input_batch)
                # TODO Check if the regularization term is correct
                loss = - F.cross_entropy(outputs, l) + self.hyperparameters["t_Regularization_term"] if "t_Regularization_term" in self.hyperparameters else 0.25 * torch.mean(torch.sum(torch.square(input_batch))) # Hyperparameter
                optimizers.zero_grad()
                loss.backward()
                optimizers.step()

                # for logging
                total_loss.append(loss.cpu().detach().numpy())

            # scheduler.step()
            if self.logs:
                print("Epoch: {}, Loss: {}".format(epoch, np.mean(total_loss)))

            if epoch >= num_epochs:
                # TODO
                # Does the loss have to be less than 0?
                break
        noise_generator.freeze()
        return noise_generator, og_labels

    def noise_maximizer(self) -> Dataset:
        
        noise_generator, original_labels = self.noise_maximization()

        return NoiseDataset(noise_generator, original_labels, len(self.data_forget))
    
