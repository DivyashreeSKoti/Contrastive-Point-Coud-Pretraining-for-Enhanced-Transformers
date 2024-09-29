import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from config.bodyscan_config import model_params
from torch.optim.lr_scheduler import StepLR

# Training loop
def train_model(model, train_dataloader, val_dataloader, device, num_epochs=250, wg_flag=True, age_flag=False):
    # Define loss functions based on tasks
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    model.to(device)
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_total_batches = len(train_dataloader)
    val_total_batches = len(val_dataloader)

    if age_flag:
        step_size = 60
        gamma = 0.1
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(num_epochs):
        train_loss_total, total_train_correct, total_train_samples = 0.0, 0, 0
        model.train()

        # Training loop
        for batch_data, batch_targets in train_dataloader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            
            train_outputs = model(batch_data)
            
            # Multi-class classification: CrossEntropy Loss, softmax activation
            train_loss = criterion(train_outputs, batch_targets.long())
            _, train_predicted = torch.max(train_outputs.data, 1)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            correct = (train_predicted == batch_targets).sum().item()
            train_loss_total += train_loss.item()
            total_train_correct += correct
            total_train_samples += batch_data.size(0)
            # Print progress
            if not wg_flag:
                print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{train_total_batches}, Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}", end="")


        # Compute average training loss and accuracy
        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        if wg_flag:
            # Validation
            val_loss_total, total_val_correct, total_val_samples = 0.0, 0, 0
            model.eval()

            with torch.no_grad():
                for batch_data, batch_targets in val_dataloader:
                    batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

                    val_outputs = model(batch_data)
                    
                    # Multi-class classification: CrossEntropy Loss, softmax activation
                    val_loss = criterion(val_outputs, batch_targets.long())

                    val_loss_total += val_loss.item()
                    correct = (val_predicted == batch_targets).sum().item()
                    total_val_correct += correct
                    total_val_samples += batch_data.size(0)

            # Compute average validation loss and accuracy
            avg_val_loss = val_loss_total / val_total_batches
            avg_val_accuracy = total_val_correct / total_val_samples
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Train Accuracy: {avg_train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, "
                  f"Validation Accuracy: {avg_val_accuracy:.4f}")
        
        if age_flag:
            scheduler.step()

    if wg_flag:
        return train_losses, train_accuracies, val_losses, val_accuracies
    else:
        return get_probabilities(model, val_dataloader, device, wg_flag=wg_flag, age_flag=age_flag)

def get_probabilities(model, val_dataloader, device, wg_flag = True, age_flag = False):
    model.eval()
    true_labels = []
    predicted_probs = []

    for batch_data, batch_targets in val_dataloader:
        true_labels.extend(batch_targets.long().tolist())
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass - Validation
        with torch.no_grad():
            # Multi-class classification: CrossEntropy Loss, softmax activation
            val_probs = torch.softmax(val_outputs, dim=1)
            predicted_probs.extend(val_probs.tolist())
           
    if wg_flag:
        # Start - With help of chatGPT
        # Zip the predicted_probs matrix with the true labels
        zipped = zip(predicted_probs, true_labels)

        # Sort the zipped list based on the true labels
        sorted_zipped = sorted(zipped, key=lambda x: x[1])

        # Unzip the sorted list to separate the sorted predicted_probs and sorted true_labels
        sorted_predicted_probs, sorted_true_labels = zip(*sorted_zipped)

        # Convert the sorted_predicted_probs back to a numpy array
        sorted_predicted_probs = np.array(sorted_predicted_probs)

        return sorted_predicted_probs
    else:
        # ***** Evaluate on the left out class
        return predicted_probs, true_labels
    
    
def get_age_probabilities(model, val_dataloader, device, num_classes):
    model.eval()
    true_labels = []
    predicted_probs = []
    predicted_labels = []
    all_probs = np.zeros((num_classes, num_classes))
    for batch_data, batch_targets in val_dataloader:
        true_labels.extend(batch_targets.long().tolist())
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        with torch.no_grad():
            val_outputs = model(batch_data)
            val_probs = torch.softmax(val_outputs, dim=1)
            _, val_predicted = torch.max(val_outputs.data, 1)

            # Convert tensor to numpy array
            val_probs_np = val_probs.cpu().numpy()
            val_predicted_np = val_predicted.cpu().numpy()
            batch_targets_np = batch_targets.cpu().numpy()

            # Update confusion matrix
            for i in range(len(batch_targets_np)):
                true_class = (batch_targets_np[i])
                predicted_class = (val_predicted_np[i])
                all_probs[true_class][predicted_class] += val_probs_np[i][predicted_class]

    # Normalize confusion matrix to get probabilities
    all_probs /= np.sum(all_probs, axis=1, keepdims=True)
    
    return all_probs
