import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from trimod_interaction.model import generate_model
from trimod_interaction.data import MultiModalDataset, NormalizeListTransform

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

def main(data_dir='../data/split', learning_rate=0.0001, rgb=True, depth=True, thermal=True, num_epochs=10):

    n_channels = 0
    if rgb:
        n_channels += 3
    if depth:
        n_channels += 1
    if thermal:
        n_channels += 1

    model = generate_model(n_channels)

    # Initialize model
    model = generate_model(input_dim=n_channels)

    # Define loss functions
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define data loaders
    transform = transforms.Compose([
        NormalizeListTransform(rgb=rgb, depth=depth, thermal=thermal),
        transforms.Resize((320, 240), antialias=None)
    ])
    train_data = MultiModalDataset(data_dir, split='train', transform=transform,
                                   # target_transform=ActionListTransform(model.selected_actions),
                                   window_size=8,
                                   rgb=rgb, depth=depth, thermal=thermal)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

    val_data = MultiModalDataset(data_dir, split='val', transform=transform,
                                 # target_transform=ActionListTransform(model.selected_actions),
                                 window_size=8,
                                 rgb=rgb, depth=depth, thermal=thermal)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

    test_data = MultiModalDataset(data_dir, split='test', transform=transform,
                                  # target_transform=ActionListTransform(model.selected_actions),
                                  window_size=8,
                                  rgb=rgb, depth=depth, thermal=thermal)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        # Validation loop
        val_loss = test(model, val_loader, criterion)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Test loop
    test_loss = test(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
