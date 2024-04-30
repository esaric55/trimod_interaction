import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from trimod_interaction.model import generate_model
from trimod_interaction.data import MultiModalDataset, NormalizeListTransform

import click

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)

    for inputs, targets in tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs= model(inputs)
        # TODO: calculate Accuracy/Precision/Recall/F1
        preds=outputs.argmax(axis=1)
        acc = accuracy(preds,targets)

        # Calculate metrics
        prec = precision(preds, targets)
        rec = recall(preds, targets)
        f1_score = f1(preds, targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # TODO: return Accuracy/Precision/Recall/F1
    # Compute average metrics
    avg_precision = precision.compute()
    avg_recall = recall.compute()
    avg_f1 = f1.compute()
    avg_accuracy = accuracy.compute()
    t_loss = total_loss/len(loader)
    return t_loss, avg_precision, avg_recall, avg_f1, avg_accuracy

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)
            # TODO: calculate Accuracy/Precision/Recall/F1
            preds=outputs.argmax(axis=1)
            acc = accuracy(preds,targets)
     
            prec = precision(preds, targets)
            rec = recall(preds, targets)
            f1_score = f1(preds, targets)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    # TODO: return Accuracy/Precision/Recall/F1
            t_loss = total_loss/len(loader)
    return t_loss, avg_precision, avg_recall, avg_f1, avg_accuracy

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
    model = model.to(device)

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
                                   targets ='actions',
                                   target_transform=torch.tensor,
                                   window_size=8,
                                   rgb=rgb, depth=depth, thermal=thermal)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)

    val_data = MultiModalDataset(data_dir, split='val', transform=transform, targets ='actions',
                                 target_transform=torch.tensor,
                                 window_size=8,
                                 rgb=rgb, depth=depth, thermal=thermal)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=2)

    test_data = MultiModalDataset(data_dir, split='test', transform=transform,
                                  # target_transform=ActionListTransform(model.selected_actions),
                                  window_size=8,
                                  rgb=rgb, depth=depth, thermal=thermal)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=2)
    
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    train_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics = {'loss': []}

    # Training loop
    for epoch in range(num_epochs):
        # TODO: store training metrics
        train_loss, avg_precision, avg_recall, avg_f1, avg_accuracy = train(model, train_loader, optimizer, criterion, device)
        
        # Store training metrics
        accuracy_values.append(avg_accuracy)
        precision_values.append(avg_precision)
        recall_values.append(avg_recall)
        f1_values.append(avg_f1)
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(avg_accuracy)
        train_metrics['precision'].append(avg_precision)
        train_metrics['recall'].append(avg_recall)
        train_metrics['f1'].append(avg_f1)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")
        print(f"Epoch {epoch+1}, Accuracy: {avg_accuracy}")
        print(f"Epoch {epoch+1}, Precision: {avg_precision}")
        print(f"Epoch {epoch+1}, Recall: {avg_recall}")
        print(f"Epoch {epoch+1}, F1 Score: {avg_f1}")


        # Validation loop
        val_loss = test(model, val_loader, criterion, device)
        # TODO: store validation metrics
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_values = [avg_accuracy, avg_precision, avg_recall, avg_f1]

    # TODO: Save model if validation loss is better
    if epoch == 0 or val_loss < min(val_metrics['loss']):
          torch.save(model.state_dict(), 'best_model.pth')

    #Converting tensor to NumPy array
    train_metrics_numpy = {key: np.array(value) for key, value in train_metrics.items()}
    val_metrics_numpy = {key: np.array(value) for key, value in val_metrics.items()}

    # Convert NumPy arrays to JSON-serializable format
    train_metrics_json = {key: value.tolist() for key, value in train_metrics_numpy.items()}
    val_metrics_json = {key: value.tolist() for key, value in val_metrics_numpy.items()}
    
    # Save metrics to JSON file
    metrics_dict = {
    'train': train_metrics_json,
    'validation': val_metrics_json
    }
    
    with open('metrics.json', 'w') as f:
            json.dump(metrics_dict, f)
   
    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Metrics Value')
    plt.title('Model Evaluation Metrics')
    plt.show()

    # Test loop
    # TODO: load best model
    model.load_state_dict(torch.load('best_model.pth'))
    # TODO: store validation metrics
    test_loss, avg_precision, avg_recall, avg_f1, avg_accuracy = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")
    print(f"Precision: {avg_precision}")
    print(f"Recall: {avg_recall}")
    print(f"F1 Score: {avg_f1}")
    print(f"Accuracy: {avg_accuracy}")

    metrics__test_dict = {
    'test loss': test_loss,
    'precision': avg_precision,
    'recall' : avg_recall,
    'F1 Score' : avg_f1,
    'Accuracy' : avg_accuracy
    }
    
    with open('test_metrics.json', 'w') as f:
            json.dump(metrics_test_dict, f)

if __name__ == "__main__":
    main()
