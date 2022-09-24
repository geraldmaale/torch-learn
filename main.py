from calendar import EPOCH
from utils import get_data_loaders, setup_logger
from trainer import fit_model, evaluate_model
from networks import FMnistModelV0, FMnistModelV1, FMnistModelV2
import torch
from torchinfo import summary
from logger import setup_logger
logging = setup_logger(__name__)

# Run normal training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LR = 0.001
BATCH_SIZE = 64
CRITERION = torch.nn.CrossEntropyLoss()

# Get data loaders
train_loader, val_loader = get_data_loaders(train_batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE)
train_features_batch, train_labels_batch = iter(train_loader).next()
INPUT_SIZE = train_features_batch[0].shape

# Model 0
model = FMnistModelV0(input_shape=1*28*28, hidden_units=256, output_shape=10).to(DEVICE)

OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# Train model   
# fit_model(model, 
#           loss_fn=CRITERION,
#           optimizer=OPTIMIZER,
#           train_loader=train_loader,
#           test_loader=val_loader,
#           device=DEVICE,
#           config={
#               "epochs": EPOCHS,             
#               "lr": LR,
#               "batch_size": BATCH_SIZE,
#           })

# # Evaluate model
# eval = evaluate_model(model, 
#                       loss_fn=CRITERION,
#                       dataloader=val_loader,
#                       device=DEVICE)
# logging.info(eval)

# Model 1
model_1 = FMnistModelV2(input_shape=INPUT_SIZE,
                        hidden_units=64, 
                        output_shape=10,
                        batch_size=BATCH_SIZE).to(DEVICE)
summary(model_1, input_size=train_features_batch[0].unsqueeze(0).shape, device=DEVICE)

OPTIMIZER = torch.optim.SGD(model_1.parameters(), lr=LR, momentum=0.9)

# Train model   
fit_model(model_1, 
          loss_fn=CRITERION,
          optimizer=OPTIMIZER,
          train_loader=train_loader,
          test_loader=val_loader,
          device=DEVICE,
          config={
              "epochs": EPOCHS,             
              "lr": LR,
              "batch_size": BATCH_SIZE,
          })

# Evaluate model
eval = evaluate_model(model_1, 
                      loss_fn=CRITERION,
                      dataloader=val_loader,
                      device=DEVICE)
logging.info(eval)