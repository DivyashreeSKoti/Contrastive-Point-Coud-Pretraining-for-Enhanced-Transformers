#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# taken directly from chatGPT, but not used in the execution of model required for the project.
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        # Compute the contrastive loss
        batch_size = logits.size(0)
        labels = torch.arange(0, batch_size, device=logits.device)
        positives = torch.diag(logits)
        logits = logits / self.temperature

        # Calculate the numerator of the InfoNCE loss
        numerator = torch.exp(positives).unsqueeze(1)
        # Calculate the denominator of the InfoNCE loss
        denominator = torch.exp(logits).sum(dim=1, keepdim=True) - torch.exp(positives)
        
        # Calculate the InfoNCE loss
        loss = -torch.log(numerator / denominator).mean()
        return loss
    
# ContrastiveAccuracy coded with help of chatGPT
class ContrastiveAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):

        # Compare predicted labels (y_pred) with true labels (y_true)
        # and count correct predictions
        self.correct += torch.sum(torch.argmax(y_pred, dim=1) == y_true).item()

        # Update the total number of samples seen so far
        self.total += len(y_true)

    def result(self):
        # Return the computed contrastive accuracy
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset_states(self):
        # Reset the internal state of the metric
        self.correct = 0
        self.total = 0

class ContrastiveModel(nn.Module):
    def __init__(
        self,
        device,
        masked_encoder,
        unmasked_encoder,
        embed_dim = 64,
        projection_dim = 1024,
        lr = 1e-3,
        step_size = 10,
        gamma = 0.1,
        lr_scheduler_flag = False,
        loss = nn.CrossEntropyLoss(),
        **kwargs
    ):
        super(ContrastiveModel, self).__init__(**kwargs)
        self.masked_encoder = masked_encoder
        self.unmasked_encoder = unmasked_encoder
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.W_masked = nn.Linear(self.embed_dim, self.projection_dim, bias=False)
        self.W_unmasked = nn.Linear(self.embed_dim, self.projection_dim, bias=False)
        self.t = nn.Parameter(torch.tensor(2.0), requires_grad=True)
    
        self.compiled_loss = loss
        self.masked_encoder_params = self.masked_encoder.parameters()
        self.unmasked_encoder_params = self.masked_encoder.parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.lr_scheduler_flag = lr_scheduler_flag
        self.device = device

    def forward(self, inputs, training=False):
        # Get the images from input
        masked_images, unmasked_images = inputs[0], inputs[1]
       
        # Get feature embeddings
        _, masked_features = self.masked_encoder(masked_images, get_embeddings=True)
        _, unmasked_features = self.unmasked_encoder(unmasked_images, get_embeddings = True)
        # Linear projection
        masked_embeddings = self.W_masked(masked_features)
        unmasked_embeddings = self.W_unmasked(unmasked_features)
        
        # Normalize masked_embeddings
        norm_masked_embeddings = masked_embeddings / torch.norm(masked_embeddings, dim=1, keepdim=True)
        # Normalize unmasked_embeddings
        norm_unmasked_embeddings = unmasked_embeddings / torch.norm(unmasked_embeddings, dim=1, keepdim=True)
        
        # Get contrastive logits
        logits = torch.matmul(norm_masked_embeddings, norm_unmasked_embeddings.t()) * torch.exp(self.t)
        return logits
    
    # Funtion to process contrastive learning
    def train_step(self, data):
        n = data[0].shape[0]
        # Get true labels for batch
        y_true = torch.arange(n).to(self.device)
        y_pred = self(data, training=True)
        # Get loss for both batch samples
        loss_masked = self.compiled_loss(y_pred,y_true.long())
        loss_unmasked = self.compiled_loss(y_pred.transpose(0, 1),y_true.long())
        loss = (loss_masked + loss_unmasked) / 2.0
        
        # Backward pass based on contrastive loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate contrastive accuracy
        contrastive_acc_metric = ContrastiveAccuracy()
        contrastive_acc_metric.update_state(y_true, y_pred)
        contrastive_acc = contrastive_acc_metric.result()
        return loss, contrastive_acc
    
    def val_step(self, data):
        n = data[0].shape[0]
        # Get true labels for batch
        with torch.no_grad():
            y_true = torch.arange(n).to(self.device)
            y_pred = self(data, training=False)
            # Get loss for both batch samples
            loss_masked = self.compiled_loss(y_pred,y_true.long())
            loss_unmasked = self.compiled_loss(y_pred.transpose(0, 1),y_true.long())
            loss = (loss_masked + loss_unmasked) / 2.0


            # Calculate contrastive accuracy
            contrastive_acc_metric = ContrastiveAccuracy()
            contrastive_acc_metric.update_state(y_true, y_pred)
            contrastive_acc = contrastive_acc_metric.result()
        return loss, contrastive_acc
    
    def fit(self, train_subsampled_dataloader, val_subsampled_dataloader, epochs):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Track the number of batches
        subsampled_total_DLbatches = len(train_subsampled_dataloader)
        subsampled_val_DLbatches = len(val_subsampled_dataloader)

        # Define the number of training epochs
        num_epochs = epochs

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            val_total_loss = 0
            val_total_acc = 0
            # Convert the training data to PyTorch tensors
            for i, (batch_data_1,batch_data_2) in enumerate(train_subsampled_dataloader):
                batch_data_1 = torch.tensor(batch_data_1)
                batch_data_2 = torch.tensor(batch_data_2)
                self.train()
                batch_data_1 = batch_data_1.to(self.device)
                batch_data_2 = batch_data_2.to(self.device)
                train_loss, contrastive_acc = self.train_step((batch_data_1, batch_data_2))
                train_loss = train_loss.item()
                total_loss += train_loss
                total_acc += contrastive_acc

                # Print progress
                print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{subsampled_total_DLbatches}, Train Loss: {train_loss:.4f}, ContrastiveAccuracy: {contrastive_acc: .4f}", end="")

                # Unload tensor from device
                batch_data_2 = batch_data_2.cpu()
                batch_data_1 = batch_data_1.cpu()

            for i, (batch_data_1,batch_data_2) in enumerate(val_subsampled_dataloader):
                batch_data_1 = torch.tensor(batch_data_1)
                batch_data_2 = torch.tensor(batch_data_2)
                self.eval()
                batch_data_1 = batch_data_1.to(self.device)
                batch_data_2 = batch_data_2.to(self.device)
                val_loss, contrastive_acc = self.val_step((batch_data_1, batch_data_2))
                val_loss = val_loss.item()
                val_total_loss += val_loss
                val_total_acc += contrastive_acc

                # Unload tensor from device
                batch_data_2 = batch_data_2.cpu()
                batch_data_1 = batch_data_1.cpu()

            # Compute the average train and validation loss and accuracy
            avg_train_loss = total_loss/subsampled_total_DLbatches
            avg_train_acc = total_acc/subsampled_total_DLbatches
            avg_val_loss = val_total_loss/subsampled_val_DLbatches
            avg_val_acc = val_total_acc/subsampled_val_DLbatches

            # keep track of loss and accuracy for Monte carlo simulation
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            # Print final results of epoch
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{subsampled_total_DLbatches}, Train Loss: {avg_train_loss:.4f}, Train ContrastiveAccuracy: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val ContrastiveAccuracy: {avg_val_acc:.4f}", end="")
            print('')
            if self.lr_scheduler_flag:
                self.scheduler.step()

            # Empty GPU cache
            torch.cuda.empty_cache()
        return val_accuracies, train_accuracies, val_losses, train_losses