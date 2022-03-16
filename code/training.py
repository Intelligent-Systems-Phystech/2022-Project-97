import torch
import IPython

from IPython.display import clear_output
from tqdm.notebook import tqdm
from math import ceil

def train_loop(model, dataloader, loss_fn, optimizer, step=0.05):
    out = display(IPython.display.Pretty('Learning...'), display_id=True)

    size = len(dataloader.dataset) 
    len_size = len(str(size))
    train_loss, correct = 0, 0
    batches = ceil(size / dataloader.batch_size) - 1

    percentage = 0
    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch / batches > percentage or batch == batches: 
            out.update(f'[{int(percentage * size)}/{size}] Loss: {loss:>8f}')
            percentage += step

    train_loss /= batches
    correct /= size

    history['train_loss'].append(train_loss)
    history['train_acc'].append(correct)


def test_loop(model, dataloader, loss_fn):

    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    batches = ceil(size / dataloader.batch_size)

    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(dataloader, leave=False, desc="Batch #")):
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= batches
    correct /= size

    history['val_loss'].append(test_loss)
    history['val_acc'].append(correct)
    
    print(f"Validation accuracy: {(100*correct):>0.1f}%, Validation loss: {test_loss:>8f} \n")
    return history


def train():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loop(model, train_dataloader, loss_fn, optimizer)
        test_loop(model, test_dataloader, loss_fn)

        scheduler.step()
