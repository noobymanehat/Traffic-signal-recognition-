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

from Model import TrafficSignNet

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Args:
        model: PyTorch model
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model
        history: Training history
    """
    since = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Iteration")
            
            # Iterate over data
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'accuracy': torch.sum(preds == labels.data).item() / inputs.size(0)})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model if best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history, save_path='training_history.png'):
 
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved as '{save_path}'")


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
    
    # Load the dataset
    train_loader, val_loader = load_gtsrb_dataset(args.data_dir, batch_size=args.batch_size)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create the model
    model = TrafficSignNet(num_classes=args.num_classes)
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.num_epochs)
    
    # Plot the training history
    plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))
    
    # Save the model
    model_path = os.path.join(args.save_dir, 'gtsrb_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")


if __name__ == "__main__":
    # Import the dataloader
    from Dataset import load_gtsrb_dataset
    
    # Load the dataset
    data_dir = "./GTSRB/Training"  # ############################# # PLEASE CHANGE PATH
    batch_size = 64
    train_loader, val_loader ,test_loader= load_gtsrb_dataset(data_dir, batch_size=batch_size)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create the model
    num_classes = 43  # GTSRB has 43 classes (0-42)
    model = TrafficSignNet(num_classes=num_classes)
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 30
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)
    
    # Plot the training history
    plot_training_history(history)
    
    # Save the model
    torch.save(model.state_dict(), 'gtsrb_model2.pth')
    print("Model saved as 'gtsrb_model2.pth'")