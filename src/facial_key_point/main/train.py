# inbuilt packages 
import os
from PIL import Image
from tqdm import tqdm
import json
import sys

# Add project root to sys.path
sys.path.append(r'D:\online class\DeepLearning\Facial key point')
print("Current working directory is", os.getcwd())

# Data science packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# PyTorch-related packages
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Import project-specific modules
from src.facial_key_point.config.config import configuration
from src.facial_key_point.datasets.datasets import FaceKeyPointData
from src.facial_key_point.model.vgg import get_model

from src.facial_key_point.utils.utils import train, plot_curve, visualization

def main():

    # Define the save path (hardcoded to the desired location)
    saved_path = r'D:\online class\DeepLearning\Facial key point\dump\version_1'
    print(f"Saving path: {saved_path}")  # Debug: Print the saving path
    model_path = os.path.join(saved_path, 'model.pth')
    hyperparam_path = os.path.join(saved_path, 'hyperparam.json')
    train_curve_path = os.path.join(saved_path, 'train_curve.png')
    vis_result_path = os.path.join(saved_path, 'vis_result.png')

    # Create directory if it doesn't exist
    if not os.path.exists(saved_path):
        print(f"Creating directory at: {saved_path}")
        os.makedirs(saved_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = FaceKeyPointData(csv_path=configuration.get('training_data_csvpath'), split='training', device=device)
    test_dataset = FaceKeyPointData(csv_path=configuration.get('test_data_csvpath'), split='test', device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=configuration.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=configuration.get('batch_size'), shuffle=False)

    # Load the model
    model = get_model(device=device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.get('learning_rate'))

    # Train the model
    train_loss, test_loss = train(configuration.get('n_epoch'), train_dataloader, test_dataloader, model, criterion, optimizer)

    # Plot and save the training/test loss curve
    plot_curve(train_loss, test_loss, train_curve_path)
    print(f"Training curve saved at {train_curve_path}")

    # Visualize and save the result
    visualization(r'D:\online class\DeepLearning\Facial key point\face.jpg', model, vis_result_path, configuration.get('model_input_size'), device=device)
    print(f"Visualization result saved at {vis_result_path}")

    # Save the hyperparameters
    with open(hyperparam_path, 'w') as f:
        json.dump(configuration, f)
    print(f"Hyperparameters saved at {hyperparam_path}")

    # Save the model's state_dict (best practice)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == '__main__':
    main()
