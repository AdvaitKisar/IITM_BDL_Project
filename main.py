# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.manual_seed(0)

import matplotlib.pyplot as plt
import numpy as np
import time, yaml
from sklearn.model_selection import train_test_split

# Importing functions
import func as f

# Loading all params
params = yaml.safe_load(open("params.yaml"))["params"] # Params are loaded from YAML file
model_no = params["model_no"] # Model No.
model_config = params["model_config"] # Model config dictionary
num_epochs = params["num_epochs"] # No. of epochs
batch_size = params["batch_size"] # Batch size


# Loading training data
orig_train_images, orig_train_labels = f.load_data('train')
print("Training data has been loaded.")
# Creating a dataset
orig_dataset = f.ImageDataset(orig_train_images, orig_train_labels)

# Splitting training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(orig_dataset.images, orig_dataset.labels, test_size = 0.2, random_state = 42)
# Create new training and validation datasets
train_dataset = f.ImageDataset(train_images, train_labels)
val_dataset = f.ImageDataset(val_images, val_labels)
print("New training and validation datasets have been created.")

# Data loaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model is created
model = f.create_model(model_no, model_config)
model
print("\nModel has been created.")

# Criterion and optimizer for training the model
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

# Training the model
print("Training starts...\n")
cost_train_list = [] # List for storing training cost
accuracy_train_list = [] # List for storing training accuracy
accuracy_val_list = [] # List for storing validation accuracy
start_orig = time.time() # Timestamp at start
train_samples = len(train_labels) # No. of training samples
val_samples = len(val_labels) # No. of validation set samples
best_acc_val = 0

for epoch in range(1, num_epochs+1): # Iterating for each epoch
    print(f"------------------------ Epoch No. {epoch} ------------------------")
    start_epoch = time.time() # Start of epoch
    total_loss = 0 # Variable for storing total loss
    correct_train, correct_val = 0, 0 # Variable for storing count of correct predictions
    train_batch_iter, val_batch_iter = 1, 1 # Counters for iterations
    
    for images, labels in train_loader: # Iterating through the train loader to train the model
        images = images.float()
        labels = torch.tensor(labels, dtype=torch.long)
        opt.zero_grad() # Gradient of optimizer is set to zero
        outputs = model(images) # Outputs are obtained
        loss = criterion(outputs, labels) # Loss is calculated
        loss.backward() # Backward propagation is performed
        total_loss += loss.item() # Total loss is updated
        opt.step() # Optimizer updates the paramaters of the model
        correct_train += f.accuracy(outputs, labels) # Count of correct predictions is updated
        outputs.detach()
        if train_batch_iter%100 == 0:
            print(f"Training Batch No. {train_batch_iter}")
            t_val = time.time() # Intermediate timestamp for logging
            print(f"Time spent from start of this epoch: {(t_val - start_epoch):.2f} seconds.")
        train_batch_iter += 1 # Counter for training iteration is updated
        
    cost_train_list.append(total_loss) # Total loss is appended
    acc_train = (correct_train/train_samples)*100 # Accuracy of Training Dataset
    accuracy_train_list.append(acc_train) # Accuracy of Training Dataset is appended
    print(f"\nEpoch {epoch}: Accuracy on Training Data = {acc_train:.2f} %, Training Cost = {total_loss:.2f}")
    
    print(f"\n\nValidating begins for Epoch {epoch}...")
    for images, labels in val_loader: # Iterating through the val loader to validate the model
        images = images.float()
        labels = torch.tensor(labels, dtype=torch.long)
        z = model(images) # Outputs are obtained
        correct_val += f.accuracy(z, labels) # Count of correct predictions is updated
        z.detach()
        if val_batch_iter%100 == 0:
            print(f"Validating Batch No. {val_batch_iter}")
            t_val = time.time() # Intermediate timestamp for logging
            print(f"Time spent from start of this epoch: {(t_val - start_epoch):.2f} seconds.")
        val_batch_iter += 1 # Counter for training iteration is updated
        
    acc_val = (correct_val/val_samples)*100 # Accuracy of Validation Dataset
    accuracy_val_list.append(acc_val) # Accuracy of Validation Dataset is appended
    print(f"\nEpoch {epoch}: Accuracy on Validation Set = {acc_val:.2f} %")
    
    if acc_val > best_acc_val:
        torch.save(model.state_dict(), f'model_{model_no}.pth') # Model is saved
        print(f"Model for epoch no. {epoch} saved.")
        best_acc_val = acc_val
    end_epoch = time.time() # End of epoch
    print(f"\nRuntime for Epoch {epoch}: {((end_epoch - start_epoch)/60):.2f} minutes.\n\n")
end_orig = time.time() # End of all epochs
print(f"\n\nRuntime: {((end_orig - start_orig)/60):.2f} minutes.")
print("Model has been trained and saved successfully.\n")

# Plot for training cost function with epochs
plt.plot(np.arange(1, num_epochs+1, 1), cost_train_list, marker=".", markersize="8", label="Training Cost Function", color="red")
plt.xlabel('Epochs', fontsize=15)
plt.ylabel("Training Cost", fontsize=15)
plt.title('Training Cost Function', color="blue")
plt.grid(linestyle='--')
plt.tight_layout()
plt.legend()
plt.savefig(f"Training_Cost_Function_for_Model_{model_no}.jpg", dpi=500)

# Plot for dataset accuracy with epochs
plt.plot(np.arange(1, num_epochs+1, 1), accuracy_train_list, marker=".", markersize="8", label="Training Dataset Accuracy", color="red")
plt.plot(np.arange(1, num_epochs+1, 1), accuracy_val_list, marker=".", markersize="8", label="Validation Dataset Accuracy", color="green")
plt.xlabel('Epochs', fontsize=15)
plt.ylabel("Accuracy (in %)", fontsize=15)
plt.title('Accuracy Variation', color="blue")
plt.grid(linestyle='--')
plt.tight_layout()
plt.legend()
plt.savefig(f"Accuracy_Variation_for_Model_{model_no}.jpg", dpi=500)

print("Plots have been saved.")

# Best model is loaded
best_model = f.create_model(model_no, model_config)  # Create a new instance of the model class
best_model.load_state_dict(torch.load(f'model_{model_no}.pth'))
print("Best model has been loaded.")

# It is set to evaluation mode for testing
best_model.eval()

# Test dataset is loaded and dataloader object is created
test_images, test_labels = f.load_data('test')
test_dataset = f.ImageDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print("Test dataset has been loaded.")

print("Evaluation begins...\n")
# Model is evaluated on train dataset
f.evaluate_model(best_model, train_loader)
# Model is evaluated on validation dataset
f.evaluate_model(best_model, val_loader)
# Model is evaluated on test dataset
f.evaluate_model(best_model, test_loader)
print("Evaluation ends...")
print("Code executed successfully.")