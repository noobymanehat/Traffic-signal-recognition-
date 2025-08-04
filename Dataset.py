import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class GTSRBDataset(Dataset):
   
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images (parent directory of class folders).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Replace semicolons with commas in the CSV file format
        self.annotations = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform
        
        # Extract class ID from the CSV filename (e.g., "GT-00000.csv" -> "00000")
        self.class_id = os.path.basename(csv_file).split('-')[1].split('.')[0]
        
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Construct the path to the image based on your directory structure
        # Image is in GTSRB/Training/[class_id]/[filename]
        img_filename = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, self.class_id, img_filename)
        
        # Check if the image exists
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")
        
        image = Image.open(img_name).convert('RGB')
        
        # Extract ROI coordinates
        roi_x1 = self.annotations.iloc[idx, 3]
        roi_y1 = self.annotations.iloc[idx, 4]
        roi_x2 = self.annotations.iloc[idx, 5]
        roi_y2 = self.annotations.iloc[idx, 6]
        
        # Crop the image to ROI
        image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get class label
        label = self.annotations.iloc[idx, 7]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


def load_gtsrb_dataset(data_dir, batch_size=32, num_workers=4):
    """
    Load the complete GTSRB dataset
    
    Args:
        data_dir (string): Root directory of the GTSRB dataset (should contain class folders)
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    
    # List to store datasets from all classes
    all_datasets = []
    
    # Loop through all class directories
    for class_folder in sorted(os.listdir(data_dir)):
        # Check if it's a directory and follows the expected pattern (5 digits)
        if os.path.isdir(os.path.join(data_dir, class_folder)) and class_folder.isdigit() and len(class_folder) == 5:
            csv_file = os.path.join(data_dir, class_folder, f"GT-{class_folder}.csv")
            
            # Check if CSV file exists
            if os.path.exists(csv_file):
                # Create dataset for this class
                class_dataset = GTSRBDataset(
                    csv_file=csv_file,
                    root_dir=data_dir,
                    transform=train_transform
                )
                
                all_datasets.append(class_dataset)
            else:
                print(f"Warning: CSV file not found for class {class_folder}")
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    # Split into train and validation
    val_size = int(0.2 * len(combined_dataset))
    test_size = int(0.1 * len(combined_dataset))
    train_size = len(combined_dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = "/Users/arjuntomar/Desktop/PROJECT/GTSRB/Training"  # ############################# # PLEASE CHANGE PATH
    train_loader, val_loader, test_loader = load_gtsrb_dataset(data_dir)
    
    for batch in test_loader:
        print(f"[DEBUG] type: {type(batch)}, len: {len(batch)}")
        break

    # Print some statistics
    num_classes = 0
    for class_folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, class_folder)) and class_folder.isdigit():
            num_classes += 1
    
    print(f"Number of classes: {num_classes}")
    
    # Get a batch
    for images, labels in test_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels in batch: {torch.unique(labels)}")
        break