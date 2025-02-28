import torch
from torch.utils.data import TensorDataset, random_split
import os

def custom_preprocessing(x_set, y_set):

    x_mean = x_set.mean(dim=(0, 1))
    x_std = x_set.std(dim=(0, 1)) + 1e-6  # Small epsilon to prevent division by zero
    y_mean = y_set.mean(dim=(0, 1))
    y_std = y_set.std(dim=(0, 1)) + 1e-6  # Small epsilon to prevent division by zero

    return x_mean, x_std, y_mean, y_std

# Function to fit the scaler and save the statistics
def custom_scaler_fit(train_dataset):
    # Convert training dataset to tensor lists
    train_x_list = [train_dataset[i][0] for i in range(len(train_dataset))]
    train_y_list = [train_dataset[i][1] for i in range(len(train_dataset))]
    
    # Convert the lists to tensors
    train_x_tensor = torch.stack(train_x_list)
    train_y_tensor = torch.stack(train_y_list)
    
    # Get mean and std for normalization
    x_mean, x_std, y_mean, y_std = custom_preprocessing(train_x_tensor, train_y_tensor)
    
    # Save the mean and std values for later use
    torch.save({'mean': x_mean, 'std': x_std}, 'x_stats.pth')
    torch.save({'mean': y_mean, 'std': y_std}, 'y_stats.pth')
    
    return x_mean, x_std, y_mean, y_std

# Function to apply standard scaling
def custom_scaler_transform(x_set, y_set, x_mean, x_std, y_mean, y_std):
    # Standardize x_set and y_set
    x_set = (x_set - x_mean) / x_std
    y_set = (y_set - y_mean) / y_std
    return x_set, y_set

def preprocess(device, data_path=''):

    # Load the dataset from the saved .pt file
    data = torch.load(os.path.join(data_path, 'individual_graspers_linear.pt'))  # No weights_only argument
    
    # Extract input and output tensors
    x_set = torch.cat(data['inputs'], dim=0)
    y_set = torch.cat(data['outputs'], dim=0)
    # Load u_set
    u_set = torch.load(os.path.join(data_path, 'u_set_individual_linear.pth'))  # Ensure correct path
    # Concatenate u_set to x_set along the last dimension
    x_set = torch.cat([x_set, u_set], dim=-1)
    # Move tensors to the specified device
    x_set, y_set = x_set.to(device), y_set.to(device)
    # Create and return a dataset
    dataset = TensorDataset(x_set, y_set)
    # Split the dataset into training and testing sets
    split_rnd_gen = torch.Generator().manual_seed(42)
    train_size = int(len(x_set) * 0.80)
    test_size = len(x_set) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=split_rnd_gen)
    
    # Fit the custom scaler to the training dataset to get mean and std
    x_mean, x_std, y_mean, y_std = custom_scaler_fit(train_dataset)
       
    return train_dataset, test_dataset, x_mean, x_std, y_mean, y_std

def custom_scaler_transform(x, x_mean, x_std):
    processed_x = (x - x_mean) / x_std
    return processed_x

def inverse_custom_scale_transform(y, y_mean, y_std):
    # Inverse transform y_set
    processed_y = (y * y_std) + y_mean
    return processed_y
