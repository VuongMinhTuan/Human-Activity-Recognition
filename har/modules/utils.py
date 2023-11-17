import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Union, Tuple, List


# Calculate the accuracy
def accuracy(y_pred, y):
    correct = torch.eq(y_pred, y).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc



# Training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_func,
               device: torch.device):
    
    train_loss, train_acc = 0, 0
    model.to(device)

    # Start training
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim= 1).argmax(dim= 1)


        # Calculate loss and accuracy
        loss = loss_func(y_logits, y)
        train_loss += loss
        train_acc += accuracy_func(y_pred, y)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Calculate the loss and accuracy per epoch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return (train_loss, train_acc)



#  Testing step
def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               accuracy_func,
               device: torch.device):
    
    test_loss, test_acc = 0, 0

    model.to(device)

    # Start evaluating
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim= 1).argmax(dim= 1)


            # Calculate loss and accuracy
            loss = loss_func(y_logits, y)
            test_loss += loss
            test_acc += accuracy_func(y_pred, y)


        # Calculate the loss and accuracy per epoch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
        return (test_loss, test_acc)
    


# Training function
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_func: torch.nn.Module,
          device: torch.device,
          accuracy_func,
          epochs: int):
    

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    

    for param in model.features.parameters():
        param.requires_grad = False
    

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model= model,
                                           data_loader= train_dataloader,
                                           loss_func= loss_func,
                                           optimizer= optimizer,
                                           accuracy_func= accuracy_func,
                                           device= device)
        
        test_loss, test_acc = test_step(model= model,
                                        data_loader= test_dataloader,
                                        loss_func= loss_func,
                                        accuracy_func= accuracy_func,
                                        device= device)
        

        # print the training process
        if (epoch + 1) % 10 == 0:
            print(
                f"\nEpoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results




# Plotting function
def plot_curves(results: dict[str, list[float]]):
    # Get the loss values of the results dictionary (training and test)
    train_loss = torch.tensor(results['train_loss'], requires_grad= True).detach().numpy()
    test_loss = torch.tensor(results['test_loss'], requires_grad= True).detach().numpy()

    # Get the accuracy values of the results dictionary (training and test)
    train_accuracy = torch.tensor(results['train_acc'], requires_grad= True).detach().numpy()
    test_accuracy = torch.tensor(results['test_acc'], requires_grad= True).detach().numpy()

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()



# Create data loader
def create_dataLoaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int):
    
    
    # Create train and test datasets
    train_data = datasets.ImageFolder(root= train_dir, transform= transform)
    test_data = datasets.ImageFolder(root= test_dir, transform= transform)

    # Create data classes
    data_classes = train_data.classes

    # Create train and test dataloaders
    train_dataloader = DataLoader(dataset= train_data,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= num_workers,
                                  pin_memory= True)
    
    test_dataloader = DataLoader(dataset= test_data,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 pin_memory= True)
    

    return (train_dataloader, test_dataloader, data_classes)




__all__ = ["workers_handler", "tuple_handler"]



# Calculate the number of workers based on an input value.
def workers_handler(value: Union[int, float]) -> int:
    max_workers = os.cpu_count()
    match value:
        case int():
            workers = value
        case float():
            workers = int(max_workers * value)
        case _:
            workers = 0
    if not (-1 < workers < max_workers):
        raise ValueError(
            f"Number of workers is out of bounds. Min: 0 | Max: {max_workers}"
        )
    return workers




# Create a tuple with specified dimensions and values.
def tuple_handler(value: Union[int, List[int], Tuple[int]], max_dim: int) -> Tuple:
    # Check max_dim
    if not isinstance(max_dim, int) and max_dim > 1:
        raise TypeError(
            f"The 'max_dim' parameter must be an int. Got {type(max_dim)} instead."
        )
    # Check value
    if isinstance(value, int):
        output = tuple([value] * max_dim)
    else:
        try:
            output = tuple(value)
        except:
            raise TypeError(
                f"The 'value' parameter must be an int or tuple or list. Got {type(value)} instead."
            )
    if len(output) != max_dim:
        raise ValueError(
            f"The lenght of 'value' parameter must be equal to {max_dim}. Got {len(output)} instead."
        )
    return output