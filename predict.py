import argparse
import os
import random
from torchvision import datasets,transforms,models
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_size = 25088
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        input_size = 1024
    else:
        raise ValueError('Unsupported architecture.')

    model.classifier = nn.Sequential(
        nn.Linear(input_size, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, model, topk=5, device='cpu'):
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float().to(device)
    with torch.no_grad():
        output = model.forward(image)
    probabilities = torch.exp(output)
    top_probs, top_labels = probabilities.topk(topk)
    top_probs = top_probs.cpu().numpy()[0]
    top_labels = top_labels.cpu().numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[label] for label in top_labels]
    return top_probs, top_classes

if __name__ == '__main__':
    args = get_input_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    model = load_checkpoint(args.checkpoint)
    model.to(device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(args.image_path, model, args.top_k, device)
    names = [cat_to_name[str(cls)] for cls in classes]

    for name, prob in zip(names, probs):
        print(f"{name}: {prob:.3f}")
