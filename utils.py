import gc
import os
import random
import time
import json
from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import display

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10, FashionMNIST
import torch

from logger import setup_logger
logging = setup_logger(__name__)


def plot_metrics(file):
    # Validate if file exists
    if not os.path.exists(file):
        print("File not found:", file)
        return

    # Load the metric file
    metrics = pd.read_csv(file)
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    # drop all rows with NaN
    # metrics.dropna(inplace=True)

    display(metrics.dropna(axis=1, how="all").head())

    sns.relplot(data=metrics, kind="line")

    history = metrics.copy()

    # Plot the training and validation accuracy
    # plt.figure(figsize=(15,5))
    # plt.subplot(121)
    # plt.plot(history['epoch'], history['train_acc'], label='Training acc')
    # plt.plot(history['epoch'], history['val_acc'], label='Validation acc')
    # plt.title('Accuracy Plot')
    # plt.legend()
    # plt.subplot(122)
    # # plt.plot(history['train_loss'], label='Training loss')
    # plt.plot(history['epoch'], history['val_loss'], label='Validation loss')
    # plt.title('Validation Plot')
    # plt.legend()
    return history

def plot_results(hist, save=False, filepath=None):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()    
    if save:
        if filepath is None:
            filepath = "./logs/%s"
        plt.savefig(filepath % "/"+"model.png")
        
def plot_datasets(dataset:Dataset, classes=None):
    n = 10
    mn = min([dataset[i][0].min() for i in range(15)])
    mx = max([dataset[i][0].max() for i in range(15)])
    fig, ax = plt.subplots(1, n, figsize=(15, 10))
    for i in range(10):
        ax[i].imshow(np.transpose((dataset[i][0]-mn)/(mx-mn),(1,2,0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[dataset[i][1]])


def log_performance(model:torch.nn.Module, 
                    history: dict, 
                    trainloader: DataLoader, 
                    config: dict, 
                    device):
    """Log the performance of the model"""
    LOG_PATH = "./logs/%s"
    TIMESTAMP = str(int(time.time()))
    model_path = f"{model.__class__.__name__}_{TIMESTAMP}"

    if not os.path.exists(LOG_PATH % model_path):
        os.makedirs(LOG_PATH % model_path)
    
    # Save hyper-parameters to json file
    write_json(config, file_path=f"{LOG_PATH % model_path}/hyperparameters.json")

    # Export model to csv
    if history is not None:
        history_df = pd.DataFrame(history)
        history_df.to_csv(LOG_PATH % (model_path+"/"+model_path+".csv"), index=False)

        # Save plots
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.plot(history['train_acc'], label='Training acc')
        plt.plot(history['test_acc'], label='Validation acc')
        plt.legend()
        plt.subplot(122)
        plt.plot(history['train_loss'], label='Training loss')
        plt.plot(history['test_loss'], label='Validation loss')
        plt.legend()    
        plt.savefig(LOG_PATH % (model_path+"/"+model_path+".png"))

    # Save model
    if model is not None:
        # Save torch model
        torch.save(model.state_dict(), LOG_PATH % (model_path+"/"+model_path+".pth"))
        
        # Export to onnx
        input_shape = trainloader.dataset[0][0].data.unsqueeze(0).shape
        tensor = torch.randn(input_shape)
        tensor = tensor.reshape(*input_shape).to(device)
        torch.onnx.export(model, tensor, LOG_PATH % (model_path+"/"+model_path+".onnx"))
            
    # log pc info
    log_pc_info(path=LOG_PATH % model_path+"/pc_info.json")
    

def export_to_onnx_lightning(train_loader, filename, model):
    """Export the pytorch-lightning model to onnx file"""
    shape = train_loader.dataset[0][0].unsqueeze(0).shape
    input_dim = tuple(shape)
    onnx_file = f"./logs/{filename}/{filename}.onnx"
    model.to_onnx(onnx_file, input_sample=torch.randn(input_dim), export_params=True)


def get_input_dim(train_loader: DataLoader):
    """Get input dimension of the model"""
    # input_shape = train_loader.dataset[0][0].unsqueeze(0).shape
    input_shape = train_loader.dataset[0][0].shape
    input_dim = tuple(input_shape)
    return input_dim


def file_path(model_name):
    TIMESTAMP = str(int(time.time()))
    model_name = f"{model_name}_{TIMESTAMP}"
    return model_name

def plot_convolution(t, data_train, title=''):
    with torch.no_grad():
        c = torch.nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1)
        c.weight.copy_(t)
        fig, ax = plt.subplots(2,6,figsize=(8,3))
        fig.suptitle(title,fontsize=16)
        for i in range(5):
            im = data_train[i][0]
            ax[0][i].imshow(im[0])
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0])
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        ax[0,5].imshow(t)
        ax[0,5].axis('off')
        ax[1,5].axis('off')
        #plt.tight_layout()
        plt.show()

def clear_cuda_cache(model):
    """Garbage collect and free GPU resources"""
    del model
    gc.collect()
    torch.cuda.empty_cache()

def load_cifar10_data(batch_size:int, num_workers:int=4):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    train_dataset = CIFAR10("./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10("./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def dataloader(batch_size):
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transforms)
    valset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=test_transforms)


    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        
    val_loader = DataLoader(valset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    input_shape = train_loader.dataset[0][0].unsqueeze(0).shape
    input_dim = tuple(input_shape)

    return train_loader, val_loader, input_dim

def export_to_onnx(model, filename, input_shape, device):
    tensor = torch.randn(input_shape)
    tensor = tensor.reshape(1, *input_shape).to(device)
    torch.onnx.export(model, tensor, filename)
    
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Plot linear data or training and test and predictions (optional)
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots linear training data and test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    
def plot_transformed_images(image_paths: list, transforms, num_images=25, seed=None):
    """
    Plots original and transformed images.
    """
    if seed:
        random.seed(seed)
    
    random_image_paths = random.sample(image_paths, k=num_images)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)
            
            # Transform image and plot target image
            transformed_image = transforms(f).permute(1, 2, 0) # Convert from CxHxW to HxWxC
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nSize: {transformed_image.size()}")
            ax[1].axis("off")
            
            fig.tight_layout()
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            
            
def print_train_time(start, end, device):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    
    print()
    logging.info(f"Training time on {device}: {total_time:.2f} seconds")
    
    return total_time


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    fmnist_train = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=data_transform,
    )

    fmnist_test = FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=data_transform
        )

    train_loader = DataLoader(fmnist_train, batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(fmnist_test, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        file_dict = pickle.load(fo, encoding='latin1')
    return file_dict


def load_json(file_path):
    """ Load a json file object """
    with open(file_path, 'r+') as file:
        file_data = json.load(file)
    return file_data


def write_json(json_dict, file_path):
    with open(file_path, 'w') as file:
        json.dump(json_dict, file, indent=4)


def sanity_check(df):
    from pandas.api import types
    """ Check for inconsistencies in data types """
    counter = 0
    for col in df.columns:
        if types.is_string_dtype(df[col]) or types.is_object_dtype(df[col]):
            print(f"Col {col} is still a string")
            counter += 1
        if df[col].isnull().any():
            print(f"Col {col} still has missing values")
            counter += 1

    if counter == 0:
        print('All columns are clean and ready for training ...')
    return counter


def datetime_gen(start_date_time, end_date_time):
    start = pd.to_datetime(start_date_time)
    end = pd.to_datetime(end_date_time)
    datetime_data = pd.date_range(start, end, freq='H')
    return datetime_data


def split_dates(dataset):
    dataset["year"] = dataset.index.year
    dataset["month"] = dataset.index.month
    dataset["day"] = dataset.index.day
    dataset["week_day"] = dataset.index.dayofweek
    dataset["hour"] = dataset.index.hour
    # dataset[colname] = dataset[colname].astype(np.int64) # convert to seconds since 1970
    return dataset


def encode_categorical(dataset):
    # sanity check
    result = sanity_check(dataset)
    if result > 0:
        print("Some columns are still strings, please convert to categorical")
        return
    cat_columns = dataset.select_dtypes('category').columns
    dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes + 1)
    return dataset


def convert_to_categorical(dataset, columns: list):
    # Convert columns to categoricals
    for col in columns:
        dataset[col] = dataset[col].astype('category')
    dataset.info()
    return dataset

def log_pc_info(path):
    """ Log PC info to a file """
    pc_env = {
        "gpu": os.popen('nvidia-smi --query-gpu=gpu_name --format=csv').read().split('\n')[1].strip(),
        "cpu": os.popen('cat /proc/cpuinfo | grep "model name" | uniq').read().split(':')[1].strip(),
        "ram": os.popen('cat /proc/meminfo | grep MemTotal').read().split(':')[1].strip(),
        "os": os.popen('cat /etc/os-release | grep PRETTY_NAME').read().split('=')[1].lstrip('"').rstrip('"\n')
    }
    write_json(pc_env, file_path=path)
    
def plot_confusion_matrix(true_labels, predicted_labels:torch.Tensor, class_names:list, filename="confusion_matrix_report.png"):
    """ Plot confusion matrix """
    from mlxtend.plotting import plot_confusion_matrix
    from torchmetrics import ConfusionMatrix
    from sklearn.metrics import classification_report
    
    confmat = ConfusionMatrix(num_classes=len(class_names))
    cm_tensor = confmat(predicted_labels, true_labels)
    
    fig, ax = plot_confusion_matrix(conf_mat=cm_tensor.numpy(), 
                                    show_absolute=True,
                                    show_normed=False,
                                    colorbar=True,
                                    figsize=(10, 10),
                                    class_names=class_names)   
        
    if filename is not None:
        fig.savefig(filename)        
    # plt.show() 