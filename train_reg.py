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
from KelvinletPhysics import *
from data_generator import *
from train_utils import feature_transform_regularizer, deformation_capture_mean
import pyvista as pv
from data_preprocessing import preprocess
from pointnet import PointNetfeat


def train_test(save_name, device, num_epochs=511, lr=5e-4, weight_decay=5e-5, opt='adam', milestones=[150, 300, 400], gamma=0.1, batch_size=16):

    # Enable multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    assert batch_size % num_gpus == 0, "Batch size must be divisible by number of GPUs"

    # Preprocess and get the datasets and scaler parameters
    train_dataset, test_dataset, x_mean, x_std, y_mean, y_std = preprocess(device=device)
    x_mean, x_std, y_mean, y_std = [t.to(device, dtype=torch.float32) for t in (x_mean, x_std, y_mean, y_std)]
    
    # Load mesh data
    mesh_nodes = torch.from_numpy(np.load('coords.npy')).to(device, dtype=torch.float32)
    coords_expanded = mesh_nodes.unsqueeze(0).expand(batch_size, -1, -1)
    
    kp = KelvinletPhysics(mesh_nodes.cpu().numpy())  # Keeping in CPU for now
    
    surface_mesh = pv.read("surface_mesh.vtk")
    generator = MeshDisplacementGenerator(
        surface_mesh.point_data['coordinates'],
        surface_mesh.point_data['Normals'],
        surface_mesh.point_data['Map to 10K']
    ) 

    # Initialize model and wrap with DataParallel
    model = PointNetfeat()
    model = nn.DataParallel(model)  # Parallelize across multiple GPUs
    model.to(device)

    # Loss criterion
    criterion = nn.MSELoss()

    # Optimizer setup
    optimizer_cls = Adam if opt == 'adam' else SGD
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Track metrics
    epochMetrics = defaultdict(list)

    # Data loaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        phase = 'train'
        log_def_mean = []

        for x, y in loader_train:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)

            # Generate displacement batch
            x_test, flags_test = generator.generate_batch(batch_size)
            x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
            flags_test = torch.tensor(flags_test, device=device, dtype=torch.float32).unsqueeze(-1)

            pred_2 = [
                kp.getSolution(
                    mesh_nodes[flags_test[b].squeeze(-1) > 0].cpu().numpy(),  # Fix mask shape
                    x_test[b, flags_test[b].squeeze(-1) > 0].cpu().numpy()    # Fix mask shape
                ) 
                for b in range(batch_size)
            ]

            y_pred2 = torch.from_numpy(np.array(pred_2)).to(device, dtype=torch.float32)
            
            # Prepare x_test
            x_test = torch.cat([coords_expanded, torch.zeros_like(coords_expanded), x_test, flags_test], dim=-1)
            x_test = (x_test - x_mean[:10]) / x_std[:10]

            # Forward pass
            yhat_pred2, _ = model(x_test)
            y_pred2 = (y_pred2 - y_mean) / y_std

            # Process original batch
            x = (x - x_mean) / x_std  # Normalize x
            y = (y - y_mean) / y_std  # Normalize y
            y_pred1, trans_feat = model(x[:, :, :10])

            # Compute losses
            loss1 = criterion(y_pred1, y) + 0.01 * feature_transform_regularizer(trans_feat, device=device)
            loss2 = criterion(yhat_pred2, y_pred2)
            loss = loss1 + loss2

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics calculation
            y_inv = y * y_std + y_mean
            y_pred_inv = y_pred1 * y_std + y_mean
            log_def_mean.append(deformation_capture_mean(y_pred_inv, y_inv))

        # Aggregate training metrics
        epochMetrics[f'{phase}_def_avg'].append(np.mean(log_def_mean))

        # Evaluation phase
        model.eval()
        phase = 'test'
        log_def_mean = []

        with torch.no_grad():
            for x, y in loader_test:
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
                x = (x - x_mean) / x_std
                y = (y - y_mean) / y_std

                y_pred, trans_feat = model(x[:, :, :10])
                y_inv = y * y_std + y_mean
                y_pred_inv = y_pred * y_std + y_mean
                log_def_mean.append(deformation_capture_mean(y_pred_inv, y_inv))
        
        # Aggregate test metrics
        epochMetrics[f'{phase}_def_avg'].append(np.mean(log_def_mean))

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Def. Avg: {epochMetrics['train_def_avg'][-1]:.4f}")
            print(f"  Test Def. Avg:  {epochMetrics['test_def_avg'][-1]:.4f}")

    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    train_test(args.save_name, args.device)
