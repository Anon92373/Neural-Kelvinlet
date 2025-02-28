import torch
import torch.nn as nn


def cuda2numpy(Tensor, device):
    return Tensor.detach().cpu().numpy()

def df2cuda(df):
    return torch.from_numpy(np.array(df)).to(device)

def deformation_capture_mean(predicted, ground_truth):

    error_vectors = predicted - ground_truth
    error_magnitudes = torch.norm(error_vectors, dim=1)
    d_error = torch.mean(error_magnitudes)
    measured_magnitudes = torch.norm(ground_truth, dim=1)
    d_meas = torch.mean(measured_magnitudes)
    percentage_correction = 100 * (1 - (d_error / d_meas))
    
    return percentage_correction.item()

def feature_transform_regularizer(trans, device):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss