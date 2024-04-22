import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.utils.data import DataLoader

from trimod_interaction.model import generate_model
from trimod_interaction.data import MultiModalDataset, NormalizeListTransform

import click

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs= model(inputs)
        # TODO: calculate Accuracy/Precision/Recall/F1

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # TODO: return Accuracy/Precision/Recall/F1
    return total_loss / len(loader)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)
            # TODO: calculate Accuracy/Precision/Recall/F1

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    # TODO: return Accuracy/Precision/Recall/F1
    return total_loss / len(loader)

@click.command()
@click.option('--data-dir', default='../data/split', help='Directory path for data')
@click.option('--learning-rate', default=0.0001, help='Learning rate for training')
@click.option('--rgb/--no-rgb', default=True, help='Include RGB data or not')
@click.option('--depth/--no-depth', default=True, help='Include depth data or not')
@click.option('--thermal/--no-thermal', default=True, help='Include thermal data or not')
@click.option('--num-epochs', default=10, help='Number of training epochs')
def main(data_dir, learning_rate, rgb, depth, thermal, num_epochs):
    # python -m trimod_interaction.cli.train --data-dir ../data/split2 --learning-rate 0.0001 --rgb --no-depth --thermal --num-epochs 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_channels = 0
    if rgb:
        n_channels += 3
    if depth:
        n_channels += 1
    if thermal:
        n_channels += 1

    # Initialize model
    model = generate_model(input_dim=n_channels)
    model.to(device)

    # Define loss functions
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define data loaders
    transform = transforms.Compose([
        NormalizeListTransform(rgb=rgb, depth=depth, thermal=thermal),
        transforms.Resize((320, 240), antialias=None)
    ])
    # target_transform = transforms.Compose([
    #     torch.tensor
    # ])
    train_data = MultiModalDataset(data_dir, split='train', transform=transform,
                                   target_transform=torch.tensor,
                                   window_size=8,
                                   rgb=rgb, depth=depth, thermal=thermal)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)

    val_data = MultiModalDataset(data_dir, split='val', transform=transform,
                                 target_transform=torch.tensor,
                                 window_size=8,
                                 rgb=rgb, depth=depth, thermal=thermal)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=4)

    test_data = MultiModalDataset(data_dir, split='test', transform=transform,
                                  # target_transform=ActionListTransform(model.selected_actions),
                                  window_size=8,
                                  rgb=rgb, depth=depth, thermal=thermal)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=4)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        # TODO: store training metrics
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        # Validation loop
        val_loss = test(model, val_loader, criterion, device)
        # TODO: store validation metrics
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # TODO: Save model if validation loss is better

    # Test loop
    # TODO: load best model
    test_loss = test(model, test_loader, criterion, device)
    # TODO: store validation metrics
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
