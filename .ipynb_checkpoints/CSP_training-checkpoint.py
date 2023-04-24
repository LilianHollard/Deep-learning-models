import torch
from CSPDarknet_YOLOv5_test import *
import requests
import zipfile
from pathlib import Path
import numpy

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from tqdm.auto import tqdm

device = "cuda:4"

def download_dataset():
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path)

            
import os
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str or pathlib.Path): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def main():
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    
    download_dataset()
    walk_through_dir(image_path)
    

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"


    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    # Use ImageFolder to create dataset(s)

    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    # Turn train and test Datasets into DataLoaders
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=1, # how many samples per batch?
                                  num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=1, 
                                 num_workers=1, 
                                 shuffle=False) # don't usually need to shuffle testing data

    img, label = next(iter(train_dataloader))

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")
    
    CSP = CSPDarknet(3, 16).to(device)
    
    img_batch, label_batch = next(iter(train_dataloader))
    img_single, label_single = img_batch[0].unsqueeze(dim=0).to(device), label_batch[0]
    
    CSP.eval()
    with torch.inference_mode():
        pred = CSP(img_single)
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")
    
    NUM_EPOCHS = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=CSP.parameters(), lr=0.001)
    
    res = train(model=CSP, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
               optimizer=optimizer,
               loss_fn=loss_fn,
               epochs=NUM_EPOCHS)
    
    
if __name__ == "__main__":
    main()