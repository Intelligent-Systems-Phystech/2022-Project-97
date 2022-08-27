from operator import mod
from pathlib import Path 

from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import IPython
from IPython.display import clear_output


from consts import batch_size, data_path, num_workers, device, use_colab, local_path, colab_path


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,))
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = FashionMNIST(
        root=data_path, train=True, download=True,
        transform=transform
    )

    test_data = FashionMNIST(
        root=data_path, train=False, download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_data, shuffle=False,
        batch_size=batch_size, num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_data, shuffle=False,
        batch_size=batch_size, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


def train_loop(model, history, mask,  dataloader, loss_fn, optimizer):

    size = 0
    train_loss, correct = 0, 0
    batches = 0

    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
        X, y = X.to(device), y.to(device)

        pred = model(X) * mask
        mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
        y, pred = y[mask_idx], pred[mask_idx]

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size += len(y)
        batches += 1

    train_loss /= batches
    correct /= size

    history['train_loss'].append(train_loss)
    history['train_acc'].append(correct)

    return history


def test_loop(model, history, mask, dataloader, loss_fn):
    size = 0
    test_loss, correct = 0, 0
    batches = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
            X, y = X.to(device), y.to(device)

            pred = model(X) * mask
            mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
            y, pred = y[mask_idx], pred[mask_idx]

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(y)
            batches += 1

    test_loss /= batches
    correct /= size

    history['val_loss'].append(test_loss)
    history['val_acc'].append(correct)

    print(
        f"Validation accuracy: {(100*correct):>0.1f}%, Validation loss: {test_loss:>8f} \n")
    return history


def get_path():
    if use_colab:
        from google.colab import drive
        drive.mount('/content/drive')
        p = Path(colab_path)
    else:
        p = Path(local_path)
    
    if not p.exists():
        p.mkdir(parents=True)
    return str(p)

class MLP(nn.Module):
    def __init__(self, blocks, in_features=28*28, n_classes=10,  bias=True):
        super().__init__()

        in_features = [in_features, *blocks]
        out_features = [*blocks, n_classes]
        
        self.flatten = nn.Flatten()

        self.stack = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features[idx], out_features[idx], bias=bias),
                nn.ReLU(),
            )
            for idx in range(len(blocks) + 1)
        ])
        
        
    def forward(self, X):
        return self.stack(self.flatten(X))

def make_teacher_model(bias=True):
    return  MLP(blocks=[128, 64, 32], bias=bias).to(device)

def make_student_model(bias=True):
    return  MLP(blocks=[256, 128, 64], bias=bias).to(device)


def test_loop_noise(model, history, mask, dataloader, epses):
    
    batches = 0
    original_params = []
    for p in model.parameters():
        original_params.append(p.data * 1.0)
    if 'param_noise_acc' not in history:
        history['param_noise_acc'] = []

    history['param_noise_acc'].append([])
    with torch.no_grad():
        
        for eps in epses:
            correct = 0
            size = 0
            for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
                X, y = X.to(device), y.to(device)
                for p_old, p in zip(original_params, model.parameters()):
                    p.data *= 0
                    p.data += p_old + torch.randn(p.data.shape) * eps 
                    
                pred = model(X) * mask
                mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
                y, pred = y[mask_idx], pred[mask_idx]

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                size += len(y)
                batches += 1

            correct /= size
            history['param_noise_acc'][-1].append(correct)
    print ('Noise Accuracy', history['param_noise_acc'][-1])

    return history

def test_loop_fsgm(model, history, mask, dataloader, loss_fn, epses):
    
    batches = 0
    original_params = []
    for p in model.parameters():
        original_params.append(p.data * 1.0)
    if 'fsgm_noise_acc' not in history:
        history['fsgm_noise_acc'] = []

    history['fsgm_noise_acc'].append([])

    for eps in epses:
        correct = 0
        size = 0
        for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
            X, y = X.to(device), y.to(device)
            model.zero_grad()
            X.requires_grad = True 
            pred = model(X) * mask
            loss = loss_fn(pred, y)
            loss.backward()
            data_grad = X.grad.data 
            X2 = X + eps * torch.sign(data_grad)
            pred = model(X2) * mask 

            mask_idx = torch.as_tensor([bool(mask[elem]) for elem in y])
            y, pred = y[mask_idx], pred[mask_idx]

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(y)
            batches += 1

        correct /= size
        history['fsgm_noise_acc'][-1].append(correct)
    print ('FSGM Accuracy', history['fsgm_noise_acc'][-1])

    return history
