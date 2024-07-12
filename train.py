import argparse
import os
import random
from torchvision import datasets,transforms,models
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_data_loaders(data_dir, batch_size=64, subset_size=None):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    if subset_size:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        indices = indices[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, validloader

def train_model(args):
    print("Starting training...")
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_size = 25088
    elif args.arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        input_size = 1024
    else:
        raise ValueError('Unsupported architecture. Choose either vgg16 or densenet121.')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(input_size, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    trainloader, validloader = get_data_loaders(args.data_dir, subset_size=1000)

    model.to(device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}..")
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                loss = criterion(logps, labels)
                validation_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {validation_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': trainloader.dataset.dataset.class_to_idx,  # Adjusted for subset
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }

    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
    print("Model trained and saved successfully.")

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory of training data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_input_args()
    train_model(args)
