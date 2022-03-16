import torch

import numpy as np
import matplotlib.pyplot as plt

from math import ceil, sqrt

def show_images(images, labels=None, title=None, transform=None, figsize=(12, 12)):
    fig = plt.figure(figsize=figsize, linewidth=5)
    grid_val = ceil(sqrt(len(images)))
    grid_specs = plt.GridSpec(grid_val, grid_val)
    
    for i, image in enumerate(images):
        ax = fig.add_subplot(grid_specs[i // grid_val, i % grid_val])
        ax.axis('off')
        
        if transform is not None:
            image = transform(image)
        
        if labels is not None:
            ax_title = labels[i]
        else:
            ax_title = '#{}'.format(i+1)
            
        ax.set_title(ax_title)
        ax.imshow(image, cmap='gray')
            
    if title:
        fig.suptitle(title, y=0.93, fontsize='xx-large')
    plt.show()

def plot_history():
    x = np.arange(1, epochs * num_repeats + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
    ax1.plot(x, history['train_loss'], label='Train')
    ax1.plot(x, history['val_loss'], label='Val')
    ax1.set_title('Loss', fontsize=16)
    ax1.legend(fontsize=14)

    ax2.plot(x, history['train_acc'], label='Train')
    ax2.plot(x, history['val_acc'], label='Val')
    ax2.set_title('Accuracy', fontsize=16)
    ax2.legend(fontsize=14)

    fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=14)

    plt.show()

def plot_variance(histories, labels, field_name, epoch_size, xlabel='', ylabel='', filename='file'):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 24

    x = np.arange(1, epoch_size + 1)

    plt.figure(figsize=(12, 10))

    for history, label in zip(histories, labels):
        data = torch.Tensor(history[field_name]).reshape(-1, epoch_size)
      
        plt.plot(x, torch.median(data, axis=0).values, label=label)

        q25 = torch.quantile(data, 0.25, dim=0)
        q75 = torch.quantile(data, 0.75, dim=0)

        plt.fill_between(x, q25, q75, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'{filename}.eps', transparent=True)
    plt.savefig(f'{filename}.png')
    plt.show()