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
num_classes = 360
batch_size = 32
epochs = 100
learning_rate = 0.0001

def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


def MAEeval(preds, labels):
    """
    Calculate MAE and ACC
    :param preds: Model predictions (class indices)
    :param labels: Ground truth labels (class indices)
    :return: Mean Absolute Error (MAE) and Accuracy (ACC, within a threshold of 5°)
    """
    errors = []
    for pred, label in zip(preds, labels):
        ang_error = angular_distance_compute(pred.item(), label.item())  # Calculate angular error
        errors.append(ang_error)

    # Calculate MAE and ACC
    mae = np.mean(errors)  # Mean Absolute Error
    acc = np.mean([error <= 5 for error in errors])  # Accuracy within a threshold of 5 degrees
    return mae, acc

# Data transformations and augmentations
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
class SimulatedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        pattern = re.compile(r'sound_(\d+)_t60_(\d+\.\d+)_snr_(\d+)_source_(\d+)_mic_(\d+)\.png')
        for azimuth_dir in os.listdir(data_dir):
            azimuth_path = os.path.join(data_dir, azimuth_dir)
            if os.path.isdir(azimuth_path):
                image_groups = {}
                for filename in os.listdir(azimuth_path):
                    if filename.endswith('.png'):
                        match = pattern.match(filename)
                        if match:
                            sound, t60, snr, azimuth, mic_id = match.groups()
                            mic_num = int(mic_id) - 1
                            key = f"{sound}_{t60}_{snr}_{azimuth}"
                            file_path = os.path.join(azimuth_path, filename)
                            if key not in image_groups:
                                image_groups[key] = [None] * 3
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
        #images = torch.stack(loaded_images)
        images = torch.cat(loaded_images, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return images, label

class RealDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        pattern = re.compile(r'sound_(\d+)_source_(\d+)_mic_(\d+)\.png')
        for azimuth_dir in os.listdir(data_dir):
            azimuth_path = os.path.join(data_dir, azimuth_dir)
            if os.path.isdir(azimuth_path):
                image_groups = {}
                for filename in os.listdir(azimuth_path):
                    if filename.endswith('.png'):
                        match = pattern.match(filename)
                        if match:
                            sound, azimuth, mic_id = match.groups()
                            mic_num = int(mic_id) - 1
                            key = f"{sound}_{azimuth}"
                            file_path = os.path.join(azimuth_path, filename)
                            if key not in image_groups:
                                image_groups[key] = [None] * 3
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
        #images = torch.stack(loaded_images)
        images = torch.cat(loaded_images, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return images, label

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ConvBnAct(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.ReLU):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = activation(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#
# class AttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim),
#         )
#
#     def forward(self, x):
#         x_res = x
#         x = self.norm1(x)
#         x, _ = self.attn(x, x, x)
#         x = x + x_res
#
#         x_res = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#         x = x + x_res
#         return x
#
#
#
# class CoAtNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.s0 = nn.Sequential(
#             ConvBnAct(9, 64, stride=2),
#             DepthwiseSeparableConv(64, 128, stride=2),
#         )
#         self.s1 = nn.Sequential(
#             DepthwiseSeparableConv(128, 256, stride=2),
#         )
#         self.s2 = nn.Sequential(
#             DepthwiseSeparableConv(256, 512, stride=2),
#         )
#         self.s3 = nn.ModuleList([
#             AttentionBlock(512, 4) for _ in range(2)
#         ])
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.s0(x)
#         x = self.s1(x)
#         x = self.s2(x)
#         b, c, h, w = x.size()
#         x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
#         for block in self.s3:
#             x = block(x)
#         x = x.mean(dim=1)
#         x = self.fc(x)
#         return x

import timm
import torch.nn as nn

class CoAtNet(nn.Module):
    def __init__(self, num_classes):
        super(CoAtNet, self).__init__()

        # Load the pretrained CoAtNet model
        self.base_model = timm.create_model('coatnet_1_224', pretrained=False)

        # Replace the first convolutional layer to adapt to custom input channel count
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
                9,  # Custom input channel count
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
            else:  # Otherwise, replace it directly in the root of the model
                setattr(self.base_model, first_conv_name, new_conv)
        else:
            raise AttributeError("Could not locate the first convolutional layer in the model.")

        # Replace the classification head to adapt to the custom number of classes
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
        "Train threshold15 Accuracy": train_accuracies5,
        "Val threshold15 Accuracy": val_accuracies5,
        "Train MAE": train_maes,
        "Val MAE": val_maes,
    }
    df = pd.DataFrame(data)

    # Save to an Excel file
    df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")

def save_results_to_excel1(file_path, epochs, train_losses, val_losses,
                          train_accuracies, val_accuracies,
                          train_maes, val_maes):
    # Create a DataFrame
    data = {
        "Epoch": epochs,
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Train Accuracy": train_accuracies,
        "Val Accuracy": val_accuracies,
        "Train MAE": train_maes,
        "Val MAE": val_maes,
    }
    df = pd.DataFrame(data)

    # Save to an Excel file
    df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, threshold=None):
   """
    Train the model.
    :param model: The model to be trained.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param device: Device to run on (CPU or GPU).
    :param num_epochs: Number of training epochs.
    :param threshold: Threshold (used for calculating accuracy). If None, threshold accuracy is not calculated.
    :return: Training loss, MAE, and accuracy records.
    """
    model.train()
    train_losses = []
    train_maes = []
    train_accuracies = []
    train_threshold_accuracies = []
    val_losses = []
    val_maes = []
    val_accuracies = []
    val_threshold_accuracies = []
    scaler = torch.amp.GradScaler()  # Define a scaler

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        mae = 0.0
        threshold_acc = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            with torch.amp.autocast('cuda'):  # Use mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            epoch_mae, _ = MAEeval(preds, labels)
            mae += epoch_mae

            if threshold is not None:
                _, epoch_threshold_acc = MAEeval(preds, labels)
                threshold_acc += epoch_threshold_acc

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        epoch_mae = mae / len(train_loader)
        train_maes.append(epoch_mae)

        epoch_acc = correct / total
        train_accuracies.append(epoch_acc)

        if threshold is not None:
            epoch_threshold_acc = threshold_acc / len(train_loader)
            train_threshold_accuracies.append(epoch_threshold_acc)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, "
                  f"MAE: {epoch_mae:.4f}, Accuracy: {epoch_acc:.4f}, "
                  f"Threshold-{threshold}° Accuracy: {epoch_threshold_acc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, "
                  f"MAE: {epoch_mae:.4f}, Accuracy: {epoch_acc:.4f}")
        # Validate the model
        val_loss, val_mae, val_acc, val_threshold_acc = evaluate_model(
            model, val_loader, criterion, device, threshold=threshold
        )
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_accuracies.append(val_acc)
        if threshold is not None:
            val_threshold_accuracies.append(val_threshold_acc)

    return train_losses, train_maes, train_accuracies, train_threshold_accuracies, val_losses, val_maes, val_accuracies, val_threshold_accuracies


def evaluate_model(model, val_loader, criterion, device, threshold=None):
   """
    Evaluate the model.
    :param model: The model to be evaluated.
    :param val_loader: Validation data loader.
    :param criterion: Loss function.
    :param device: Device to run on (CPU or GPU).
    :param threshold: Threshold (used for calculating accuracy). If None, threshold accuracy is not calculated.
    :return: Validation loss, MAE, accuracy records.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    mae = 0.0
    threshold_acc = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Calculate MAE and threshold accuracy
            epoch_mae, _ = MAEeval(preds, labels)
            mae += epoch_mae

            if threshold is not None:
                _, epoch_threshold_acc = MAEeval(preds, labels)
                threshold_acc += epoch_threshold_acc

    val_loss /= len(val_loader.dataset)
    val_mae = mae / len(val_loader)
    val_acc = correct / total

    if threshold is not None:
        val_threshold_acc = threshold_acc / len(val_loader)
        print(f"Validation - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, "
              f"Accuracy: {val_acc:.4f}, Threshold-{threshold}° Accuracy: {val_threshold_acc:.4f}")
    else:
        print(f"Validation - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_acc:.4f}")

    return val_loss, val_mae, val_acc, (val_threshold_acc if threshold is not None else None)

# Training and validation process
if __name__ == "__main__":
    # Load simulated dataset
    simulated_dataset = SimulatedDataset('./simulated_data', transform=data_transforms['train'])
    train_size_sim = int(0.8 * len(simulated_dataset))
    val_size_sim = len(simulated_dataset) - train_size_sim
    train_dataset_sim, val_dataset_sim = random_split(simulated_dataset, [train_size_sim, val_size_sim])
    print('simulated_dataset:',len(simulated_dataset))

    #Load real dataset
    real_dataset = RealDataset('./real_data5', transform=data_transforms['train'])
    print('real_dataset:',len(real_dataset))
    train_size_real = int(0.7 * len(real_dataset))
    val_size_real = len(real_dataset) - train_size_real
    train_dataset_real, val_dataset_real = random_split(real_dataset, [train_size_real, val_size_real])


    # Create data loaders
    train_loader_sim = DataLoader(train_dataset_sim, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_sim = DataLoader(val_dataset_sim, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader_real = DataLoader(train_dataset_real, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_real = DataLoader(val_dataset_real, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Training on simulated data...")
    train_losses, train_maes, train_accuracies, train_threshold_accuracies, \
        val_losses, val_maes, val_accuracies, val_threshold_accuracies \
            =train_model(model, train_loader_sim, val_loader_sim, criterion, optimizer, device, epochs)
    
    # Save model
    torch.save(model.state_dict(), 'coatnet_simulated_model.pth')
    # Save training results
    save_results_to_excel1(
        file_path="coatnet_simulate_results.xlsx",
        epochs=list(range(1, len(train_losses) + 1)),
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_maes=train_maes,
        val_maes=val_maes
    )

    #Fine-tune on real data model
    print("Fine-tuning on real data...")
    model.load_state_dict(torch.load('coatnet_simulated_model.pth',map_location=device,weights_only=True))

    # Choose fine-tuning strategy
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze feature extractor
    # # Assume the last two layers of the model need to be unfrozen
    # layers = list(model.children())  # Get all sub-layers
    # for layer in layers[-2:]:  # Traverse the last two layers
    #     for param in layer.parameters():  # Access each parameter of the layer
    #         param.requires_grad = True  # Unfreeze parameters


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_losses, train_maes, train_accuracies, train_threshold_accuracies, \
        val_losses, val_maes, val_accuracies, val_threshold_accuracies\
        =train_model(model, train_loader_real, val_loader_real, criterion, optimizer, device, 100, threshold=5)

    # Evaluate on real data
    # print('Evaluate on real data...')
    # evaluate_model(model, val_loader_real, criterion, device, threshold=15)

    # Save fine-tuning results
    save_results_to_excel(
        file_path="coatnet_true_results.xlsx",
        epochs=list(range(1, len(train_losses) + 1)),
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_accuracies5=train_threshold_accuracies,
        val_accuracies5=val_threshold_accuracies,
        train_maes=train_maes,
        val_maes=val_maes
    )

    # Visualize the training process
    epochs_range = range(100)
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
    plt.plot(epochs_range, train_threshold_accuracies, label='Train ACC')
    plt.plot(epochs_range, val_threshold_accuracies, label='Validation ACC')
    plt.legend()
    plt.title('Accuracy within Threshold')

    plt.tight_layout()
    plt.show()
