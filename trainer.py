import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from timeit import default_timer as timer
import time
import os
from utils import print_train_time, log_performance, plot_confusion_matrix

from logger import setup_logger
logging = setup_logger(__name__)

def fit(epochs:int, 
        model, 
        loss_fn:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        x_train, 
        y_train, 
        x_test, 
        y_test):
    
    train_accuracies = list()
    test_accuracies = list()
    train_losses = list()
    test_losses = list()    
    
    for epoch in range(epochs):
        model.train()
        
        logits = model(x_train)
        preds = torch.round(torch.sigmoid(logits))
        
        loss = loss_fn(logits.squeeze(), y_train)
        acc = accuracy_score(y_train.detach().cpu().numpy(), preds.detach().cpu().numpy())*100
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
                
        # Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_score(y_test.detach().cpu().numpy(), test_preds.detach().cpu().numpy())*100
            
        train_accuracies.append(acc)
        test_accuracies.append(test_acc)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
            
        # Print the loss and accuracy
        if epoch % 20 == 0:
            print(f"Epoch: {epoch+1} -> Loss: {loss.item():.5f}, Test Test: {test_loss.item(): 5f} | Acc: {acc:.2f}%, Test Acc: {test_acc:.2f}%")
    history = {
        "train_accuracy": train_accuracies,
        "test_accuracy": test_accuracies,
        "train_loss": train_losses,
        "test_loss": test_losses
    }
    return history

def fit_(epochs:int, 
        model, 
        loss_fn:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        x_train, 
        y_train, 
        x_test, 
        y_test,
        device):
    
    
    for epoch in range(epochs):
        # Place data on device
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        model.train()
        logits = model(x_train)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        
        loss = loss_fn(logits, y_train)
        accuracy = Accuracy().to(device)
        acc = accuracy(preds, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test)
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_test)
            accuracy = Accuracy().to(device)
            test_acc = accuracy(test_preds, y_test)
            # test_acc = accuracy_score(y_test.detach().cpu().numpy(), test_preds.detach().cpu().numpy())
        
        # Print the metrics
        if epoch % 10 == 0:
            print(f"Epoch: {epoch+1}\t-> Loss: {loss.item():.5f}, Test Loss: {test_loss.item(): 5f} | Acc: {acc:.2f}%, Test Acc: {test_acc:.2f}%")

def predict(model, x):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        preds = torch.round(torch.sigmoid(logits))
    return preds

def evaluate(model, loss_fn:torch.nn.Module, x, y):
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        preds = torch.round(torch.sigmoid(logits))
        acc = accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())*100
        loss = loss_fn(logits.squeeze(), y)
    return loss.item(), acc
            

def _train_batch_step(model, trainloader:DataLoader, optimizer, loss_fn, device):
    model.train()
    total_loss, acc = 0, 0
    pbar = tqdm(trainloader)    
    for X_train, y_train in pbar:
    # for batch, (X_train, y_train) in enumerate(trainloader):
        y_train = y_train.to(device)
        X_train = X_train.to(device)        
        
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train) #cross_entropy(out,labels)
                
        total_loss += loss.item()           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc += accuracy_score(torch.argmax(y_pred, dim=1).detach().cpu(), y_train.detach().cpu())
        # acc += accuracy(predicted, labels).item()
        # acc += accuracy_fn(y_pred, y_train)    
        
        # print(f"Batch: {batch+1}/{len(trainloader)} -> Loss: {loss.item():.5f} | Acc: {acc:.2f}%")
    return total_loss, acc

def _validation_step(model, testloader:DataLoader ,loss_fn, device):
    model.eval()
    test_acc, test_loss = 0, 0
    with torch.no_grad():
        for X_test, y_test in testloader:
            y_test = y_test.to(device)
            X_test = X_test.to(device)
            y_pred = model(X_test)
            loss = loss_fn(y_pred, y_test)
            
            test_loss += loss.item() 
            test_acc += accuracy_score(torch.argmax(y_pred, dim=1).detach().cpu(), y_test.detach().cpu())
    
    return test_loss, test_acc

def fit_model(model:torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              optimizer:torch.optim.Optimizer, 
              train_loader: DataLoader, 
              test_loader: DataLoader, 
              device, config:dict={}):
    if config is not None:
        epochs = config['epochs']

        TIMESTAMP = str(int(time.time()))

        name = f"{model.__class__.__name__}-{optimizer.__class__.__name__}-{TIMESTAMP}"
        # Initializing wandb
        # wandb.init(project=config["project_name"], name=name, config=config)
    else:
        assert False, "No config provided"
        
    # log wandb gradients, weights, and others
    # wandb.watch(models=model, criterion=loss_fn, log="parameters", log_freq=10)
    # wandb.watch(net)

    history = { 'epoch':[], 'train_loss' : [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    avg_train_loss = 0.0
    avg_test_loss = 0.0
    avg_train_acc = 0.0
    avg_test_acc = 0.0
    
    print()
    logging.info(f"Training {model.__class__.__name__} model...")
    
    torch.manual_seed(42)
    start_time = timer()
    for ep in range(epochs):
        logging.info(f"Epoch {ep+1:2}/{epochs}")
        train_loss, train_acc = _train_batch_step(model, train_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)       
        test_loss, test_acc = _validation_step(model,test_loader,loss_fn=loss_fn, device=device)

        # Average loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)     
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader) 
        
        # Log the metrics
        # logging.info(f"Epoch {ep+1:2}/{epochs} -> Train loss: {avg_train_loss:.4f} - Val loss: {avg_test_loss:.4f} | Train acc: {avg_train_acc:.2f} - Val acc: {avg_test_acc:.2f}")
        logging.info(f"\tTrain loss: {avg_train_loss:.4f} - Val loss: {avg_test_loss:.4f} | Train acc: {avg_train_acc:.2f} - Val acc: {avg_test_acc:.2f}")
        history['epoch'].append(ep+1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(avg_test_acc)

        # log to wandb
        # wandb.log({'epoch': ep+1, 'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va})   
    
    # wandb.save(file_path)
    
    # wandb.finish()
    
    # print training time
    end_time = timer()
    train_time = print_train_time(start_time, end_time, device)
    
    # log metrics
    config["model"] = model.__class__.__name__
    config["loss_fn"] = loss_fn.__class__.__name__
    config["optimizer"] = optimizer.__class__.__name__
    config['train_time'] = train_time
    config["train_loss"] = avg_train_loss
    config["train_accuracy"] = avg_train_acc
    config["test_loss"] = avg_test_loss
    config["test_accuracy"] = avg_test_acc
    config["torch_version"] = torch.__version__
    
    log_performance(model,
                    history=history,
                    trainloader=train_loader,
                    config=config,
                    device=device)    
    
    return history

def evaluate_model(model:torch.nn.Module, 
                   dataloader:DataLoader,
                   loss_fn:torch.nn.Module, 
                   device):
    torch.manual_seed(42)
    model.eval()
    running_acc, running_loss = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            y = y.to(device)
            X = X.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            running_loss += loss.item() 
            running_acc += accuracy_score(torch.argmax(y_pred, dim=1).detach().cpu(), y.detach().cpu())
            
        # Scale loss and acc by number of batches
        running_loss = running_loss / len(dataloader)
        running_acc = running_acc / len(dataloader)
        
    return {"model_name":model.__class__.__name__,
            "model_loss": "{:.4f}".format(running_loss),
            "model accuracy": "{:.2f}".format(running_acc)
            }



    
def make_sample_predictions(model:torch.nn.Module, 
                     data: list,
                     device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for feature in tqdm(data, desc="Making predictions ..."):
            feature = torch.unsqueeze(feature, dim=0).to(device)
            
            logit = model(feature)
            pred_prob = torch.softmax(logit.squeeze(), dim=0).cpu()
            predictions.append(pred_prob)            
    return predictions, torch.stack(predictions)
    
def make_predictions(model:torch.nn.Module, 
                     dataloader: DataLoader,
                     device,
                     plot_cm:bool=False,
                     class_names:list=[None]):
    predictions = []
    truth_labels = []
    
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            y = y.to(device)
            X = X.to(device)
            
            logit = model(X)
            pred_prob = torch.nn.functional.softmax(logit.squeeze(), dim=1).cpu()
            predictions.append(pred_prob)          
            truth_labels.append(y.cpu())
            
    predicted_labels, truth_labels = torch.cat(predictions), torch.cat(truth_labels)
    
    if plot_cm and len(class_names) > 1:
        plot_confusion_matrix(truth_labels, predicted_labels, class_names)
    
    return predicted_labels, truth_labels