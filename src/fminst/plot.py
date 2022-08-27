import matplotlib.pylab as plt
import numpy as np 
import torch 

def prepare_for_plots():
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 24
    plt.figure(figsize=(12, 10))
    

def plot_history(history, epochs, num_repeats):
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

def plot_variance(histories, x, labels, field_name, xlabel='', ylabel='', filename=None,
                  mode='std', y0 = 0, y1 = 1.0):
    
    
    for history, label in zip(histories, labels):
        data = np.array(history[field_name]).reshape(-1, len(x))
        if mode == 'std':
            plt.plot(x, data.mean(axis=0), label=label)
            plt.fill_between(x, data.mean(axis=0) - data.std(axis=0),
                             data.mean(axis=0) +  data.std(axis=0), alpha=0.3)
        elif mode == 'q':
            plt.plot(x, torch.median(data, axis=0), label=label)
            plt.fill_between(x, data.min(axis=0), data.max(axis=0), alpha=0.3)
    plt.ylim(y0, y1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if filename is not None:
        plt.savefig(f'{filename}.eps', transparent=True)
        plt.savefig(f'{filename}.png')
    plt.show()
    
