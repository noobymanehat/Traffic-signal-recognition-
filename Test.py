import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import DataLoader
from Model import TrafficSignNet


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to the appropriate device
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        # Iterate properly through the data loader
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # Move inputs and labels to the same device as the model
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return test_acc, predictions, true_labels

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GTSRB Traffic Sign Recognition Model')
    parser.add_argument('--data_dir', type=str, default='./GTSRB/Training', help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_classes', type=int, default=43, help='Number of classes in dataset')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save model and plots')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Import the dataloader
    try:
        from Dataset import load_gtsrb_dataset
    except ImportError:
        print("Error: Dataset.py not found in the current directory.")
        return
    
    # Load the dataset - now correctly unpacking all three loaders
    train_loader, val_loader, test_loader = load_gtsrb_dataset(args.data_dir, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TrafficSignNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load('/Users/arjuntomar/Desktop/PROJECT/gtsrb_model.pth', map_location=device))
    model = model.to(device)
    
    # Use test_loader directly, not wrapped in a dictionary
    test_acc, predictions, true_labels = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"Predictions sample: {predictions[:10]}")
    print(f"True Labels sample: {true_labels[:10]}")

if __name__ == "__main__":
    main()