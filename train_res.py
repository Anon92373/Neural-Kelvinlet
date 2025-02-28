import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from train_utils import feature_transform_regularizer

from data_preprocessing import preprocess
from pointnet import PointNetfeat


def train_test(save_name, device, num_epochs=511, lr=5e-4, weight_decay=5e-5, opt='adam', milestones=[150, 300, 400], gamma=0.1, batch_size=128):

    # Preprocess and get the datasets and scaler parameters
    train_dataset, test_dataset, x_mean, x_std, y_mean, y_std = preprocess(device=device)
    x_mean, x_std, y_mean, y_std = [t.to(device).float() for t in (x_mean, x_std, y_mean, y_std)]
    
    # Initialize model
    model = PointNetfeat().to(device)
    
    # Loss criterion
    criterion = nn.MSELoss()

    # Optimizer setup
    optimizer_cls = Adam if opt == 'adam' else SGD
    optimizer = optimizer_cls(model.parameters(), lr=lr,weight_decay = 5e-5)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Track metrics
    epochMetrics = defaultdict(list)

    # Data loaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        phase = 'train'
        
        log_def_mean, log_def_max, log_def_med = [], [], []

        # Training batch
        for x, y in loader_train:
            x, y = x.to(device).float(), y.to(device).float()
            x = (x - x_mean) / x_std  # Normalize x
            y = (y - y_mean) / y_std  # Normalize y

            y_p, trans_feat = model(x[:,:,:10].float())
            y_pred = x[:,:,10:] + y_p
    

            # Add regularization and compute total loss
            loss = criterion(y_pred,y).mean() + 0.01 * feature_transform_regularizer(trans_feat, device=device)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics calculation
            y_inv = y * y_std + y_mean  # Inverse transform y
            y_pred_inv = y_pred * y_std + y_mean  # Inverse transform y_pred

            log_def_mean.append(deformation_capture_mean(y_pred_inv, y_inv))
            log_def_max.append(deformation_capture_max(y_pred_inv, y_inv))
            log_def_med.append(deformation_capture_median(y_pred_inv, y_inv))
        
        # Aggregate metrics for training phase
        epochMetrics[f'{phase}_def_avg'].append(np.mean(log_def_mean))

        # Testing loop
        model.eval()
        phase = 'test'

        log_def_mean = []

        with torch.no_grad():
            for x, y in loader_test:
                x, y = x.to(device).float(), y.to(device).float()
                x = (x - x_mean) / x_std  # Normalize x
                y = (y - y_mean) / y_std  # Normalize y

                y_p, trans_feat = model(x[:,:,:10].float())
                y_pred = x[:,:,10:] + y_p
                # Compute metrics
                y_inv = y * y_std + y_mean  # Inverse transform y
                y_pred_inv = y_pred * y_std + y_mean  # Inverse transform y_pred

                log_def_mean.append(deformation_capture_mean(y_pred_inv, y_inv))
                log_def_max.append(deformation_capture_max(y_pred_inv, y_inv))
                log_def_med.append(deformation_capture_median(y_pred_inv, y_inv))
        
        # Aggregate metrics for testing phase
        epochMetrics[f'{phase}_def_avg'].append(np.mean(log_def_mean))

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Def Avg. acc [train]: {epochMetrics['train_def_avg'][-1]:.2f}")
            print(f"Def Avg. acc [test]: {epochMetrics['test_def_avg'][-1]:.2f}")
    
    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    
    args = parser.parse_args()
    train_test(args.save_name, args.device)
