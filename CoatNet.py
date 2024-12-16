import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
import re
import numpy as np
import pandas as pd

img_height = 224
img_width = 224
num_classes = 361
batch_size = 32
epochs = 100
learning_rate = 0.00005

def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


def MAEeval(preds, labels):
    """
    Calculate MAE and ACC
    :param preds: Model predictions (class indices)
    :param labels: True labels (class indices)
    :return: Mean Absolute Error (MAE) and Accuracy (ACC, within a 5-degree threshold)
    """
    errors = []
    for pred, label in zip(preds, labels):
        ang_error = angular_distance_compute(pred.item(), label.item())  # Compute angular error
        errors.append(ang_error)

    # Calculate MAE and ACC
    mae = np.mean(errors)  # Mean Absolute Error
    acc = np.mean([error <= 5 for error in errors])  # Accuracy within a 5-degree threshold
    return mae, acc

# Data transformations and augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataset definition
class SoundDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        pattern = re.compile(r'class(\d+)_(\d+)_(\d+)_mic(\d+)\.png')
        for azimuth_dir in os.listdir(data_dir):
            azimuth_path = os.path.join(data_dir, azimuth_dir)
            if os.path.isdir(azimuth_path):
                image_groups = {}
                for filename in os.listdir(azimuth_path):
                    if filename.endswith('.png'):
                        match = pattern.match(filename)
                        if match:
                            sound_class, azimuth, audio_id, mic_id = match.groups()
                            mic_num = int(mic_id) - 1
                            key = f"{sound_class}_{azimuth}_{audio_id}"
                            file_path = os.path.join(azimuth_path, filename)
                            if key not in image_groups:
                                image_groups[key] = [None] * 4
                            image_groups[key][mic_num] = file_path
                for key, images in image_groups.items():
                    if all(image is not None for image in images):
                        self.data.append(images)
                        self.labels.append(int(azimuth_dir.split('_')[-1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        label = self.labels[idx]
        loaded_images = []
        for image_path in images:
            with Image.open(image_path) as image:
                if self.transform:
                    image = self.transform(image)
                loaded_images.append(image)
        images = torch.cat(loaded_images, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return images, label


import timm
import torch.nn as nn

class CoAtNet(nn.Module):
    def __init__(self, num_classes):
        super(CoAtNet, self).__init__()

        # Load the pretrained CoAtNet model
        self.base_model = timm.create_model('coatnet_1_224', pretrained=False)

        # Replace the first convolutional layer to adapt to custom input channels
        # Locate the stem or conv1 layer, depending on the model structure
        first_conv_name = None
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):  # Find the first Conv2d layer
                first_conv_name = name
                break

        if first_conv_name is not None:
            # Extract the original convolutional layer
            original_conv = dict(self.base_model.named_modules())[first_conv_name]
            # Replace with a new convolutional layer
            new_conv = nn.Conv2d(
                12,  # Custom input channels
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            # Replace the module
            parent, _, attr = first_conv_name.rpartition(".")
            if parent:  # If the parent module exists
                setattr(dict(self.base_model.named_modules())[parent], attr, new_conv)
            else:  # Otherwise, replace at the model root
                setattr(self.base_model, first_conv_name, new_conv)
        else:
            raise AttributeError("Could not locate the first convolutional layer in the model.")

        # Replace the classification head to adapt to custom number of classes
        if hasattr(self.base_model, 'fc'):
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        elif hasattr(self.base_model, 'classifier'):
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)
        else:
            self.base_model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)


def save_results_to_excel(file_path, epochs, train_losses, val_losses,
                          train_accuracies, val_accuracies,
                          train_accuracies5, val_accuracies5,
                          train_maes, val_maes):
    # Create a DataFrame
    data = {
        "Epoch": epochs,
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Train Accuracy": train_accuracies,
        "Val Accuracy": val_accuracies,
        "Train threshold5 Accuracy": train_accuracies5,
        "Val threshold5 Accuracy": val_accuracies5,
        "Train MAE": train_maes,
        "Val MAE": val_maes,
    }
    df = pd.DataFrame(data)

    # Save to an Excel file
    df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")


# Training and validation process
if __name__ == "__main__":
    # Load data
    data_dir = './prepared_data'  # Replace with your data path
    dataset = SoundDataset(data_dir, transform=data_transforms['train'])

    # Check dataset length
    dataset_length = len(dataset)
    if dataset_length == 0:
        raise ValueError("Dataset is empty. Please check the data path or format.")

    # Split into training and validation sets
    train_size = int(0.7 * dataset_length)
    val_size = dataset_length - train_size

    # Ensure split sizes are valid
    if train_size == 0 or val_size == 0:
        raise ValueError("Training or validation set size is zero. Please check dataset size and split ratio.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = CoAtNet(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model parameters
    print("Model Parameters:")
    table = []
    for name, param in model.named_parameters():
        table.append([name, param.requires_grad, param.numel()])
    print(tabulate(table, headers=["Layer (type)", "Trainable", "Param #"]))

    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = torch.amp.GradScaler()  # Define a scaler

    # Train the model
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_accuracies5 = []
    val_accuracies5 = []
    train_maes = []
    val_maes = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        train_mae = 0.0
        train_acc5 = 0.0
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # Use mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            # Check for NaN loss value
            if loss.isnan().any():
                print("Loss is NaN, skipping this iteration.")
                continue

            scaler.scale(loss).backward()  # Backpropagation with scaler
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            MAE, ACC = MAEeval(preds, labels)  # DOA evaluation
            train_mae += MAE
            train_acc5 += ACC

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        train_mae /= len(train_loader)
        train_maes.append(train_mae)

        train_acc = correct / total
        train_accuracies.append(train_acc)

        train_acc5 /= len(train_loader)
        train_accuracies5.append(train_acc5)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, MAE: {train_mae:.4f}, Accuracy: {train_acc:.4f}, Accuracy(5): {train_acc5:.4f}')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_mae = 0.0
        val_acc5 = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                MAE, ACC = MAEeval(preds, labels)
                val_mae += MAE
                val_acc5 += ACC

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        val_mae /= len(val_loader)
        val_maes.append(val_mae)

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        val_acc5 /= len(val_loader)
        val_accuracies5.append(val_acc5)

        print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Accuracy: {val_acc:.4f}, Val Accuracy(5): {val_acc5:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'multi_coatnet_sound_classification.pth')
    # Save the training results
    save_results_to_excel(
        file_path="coatnet_results.xlsx",
        epochs=list(range(1, len(train_losses) + 1)),
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_accuracies5=train_accuracies5,
        val_accuracies5=val_accuracies5,
        train_maes=train_maes,
        val_maes=val_maes
    )

    # Visualize training process
    epochs_range = range(epochs)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_maes, label='Train MAE')
    plt.plot(epochs_range, val_maes, label='Validation MAE')
    plt.legend()
    plt.title('Mean Absolute Error (MAE)')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, train_accuracies5, label='Train ACC')
    plt.plot(epochs_range, val_accuracies5, label='Validation ACC')
    plt.legend()
    plt.title('Accuracy within Threshold')

    plt.tight_layout()
    plt.show()
