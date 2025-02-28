import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
# import os

from train_utils import (
    deformation_capture_mean,
    deformation_capture_max,
    deformation_capture_median,
    feature_transform_regularizer,
)
from data_preprocessing import preprocess
from pointnet import PointNetfeat

import KelvinletPhysics as Klp

def train_test( save_name, device,
                epochs = 500,
                lr = 5e-4, weight_decay = 5e-5,
                opt = 'adam', milestones = [ 150, 300, 400 ], gamma = 0.1,
                batch_size = 64,
                Kelvinlets = False, w_Kelvinlets = 1 ):
    
    # Preprocess and get the datasets and scaler parameters
    train_dataset, test_dataset, x_mean, x_std, y_mean, y_std = preprocess( device = device, data_path = 'data', aug = False )
    x_mean, x_std, y_mean, y_std = [ t.to( device ).float() for t in ( x_mean, x_std, y_mean, y_std ) ]
    
    # Initialize model
    model = PointNetfeat().to( device )
    
    # Loss criterion
    criterion = nn.MSELoss()

    # Optimizer setup
    optimizer_cls = Adam if opt == 'adam' else SGD
    optimizer = optimizer_cls( model.parameters(), lr = lr, weight_decay = weight_decay )
    scheduler = MultiStepLR( optimizer, milestones = milestones, gamma = gamma )

    # Track metrics
    epochMetrics = defaultdict(list)

    # Data loaders
    loader_train = DataLoader( train_dataset, batch_size = batch_size, shuffle = True )
    loader_test = DataLoader( test_dataset, batch_size = batch_size, shuffle = False )

    for epoch in tqdm( range( epochs ) ):
        # Training loop
        model.train()
        phase = 'train'
        
        log_def_mean, log_def_max, log_def_med = [], [], []

        # Training batch
        for x, y in loader_train:
            
            if ( Kelvinlets != False ):
                Klflags = x[ :, :, -1 ]
                Klflags_nz = torch.nonzero( Klflags )
                
                # Klflags_nnz0 = torch.count_nonzero( Klflags, dim = 0 )
                # print( Klflags_nnz0.max() )
                Klflags_nnz1 = torch.count_nonzero( Klflags, dim = 1 )
                print( Klflags_nnz1.max() )
                
                Klflags_sum0 = torch.sum( Klflags, dim = 0 )
                print( Klflags_sum0.max() )
                # Klflags_sum1 = torch.sum( Klflags, dim = 1 )
                # print( Klflags_sum1.max() )

                centers = torch.nonzero( Klflags_sum0 ).squeeze()
                print( centers.size() )

            x, y = x.to( device ).float(), y.to( device ).float()
            x = (x - x_mean) / x_std  # Normalize x
            y = (y - y_mean) / y_std  # Normalize y
            
            y_pred, trans_feat = model( x )

            if ( Kelvinlets != False ):
                xQ = x[ :, :, 0:3 ]
                x0 = x[ :, centers, 0:3 ]
                f0 = x[ :, centers, 6:9 ]
                myKlp = Klp.KelvinletPhysics( x0 )
                K, u, SE = myKlp.Kelvinlet( xQ, x0, f0, minibatch = True )
                y_pred = y_pred + ( w_Kelvinlets * u )
            
            # Compute loss with regularization
            w_reg = 0.01
            loss = criterion( y_pred, y ) + w_reg * feature_transform_regularizer( trans_feat, device = device )
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics calculation
            y_inv = y * y_std + y_mean  # Inverse transform y
            y_pred_inv = y_pred * y_std + y_mean  # Inverse transform y_pred

            log_def_mean.append( deformation_capture_mean( y_pred_inv, y_inv ) )
            log_def_max.append( deformation_capture_max( y_pred_inv, y_inv ) )
            log_def_med.append( deformation_capture_median( y_pred_inv, y_inv ) )
        
        # Aggregate metrics for training phase
        epochMetrics[ f'{phase}_def_avg' ].append( np.mean( log_def_mean ) )
        epochMetrics[ f'{phase}_def_max' ].append( np.mean( log_def_max ) )
        epochMetrics[ f'{phase}_def_median' ].append( np.mean(log_def_med ) )

        # Testing loop
        model.eval()
        phase = 'test'

        log_def_mean, log_def_max, log_def_med = [], [], []

        with torch.no_grad():
            for x, y in loader_test:

                if ( Kelvinlets != False ):
                    Klflags = x[ :, :, -1 ]
                    Klflags_agg = torch.sum( Klflags, dim = 0 )
                    centers = torch.nonzero( Klflags_agg ).squeeze()

                x, y = x.to( device ).float(), y.to( device ).float()
                x = ( x - x_mean ) / x_std  # Normalize x
                y = ( y - y_mean ) / y_std  # Normalize y

                y_pred, trans_feat = model( x.float() )

                if ( Kelvinlets != False ):
                    xQ = x[ :, :, 0:3 ]
                    x0 = x[ :, centers, 0:3 ]
                    f0 = x[ :, centers, 6:9 ]
                    myKlp = Klp.KelvinletPhysics( x0 )
                    K, u, SE = myKlp.Kelvinlet( xQ, x0, f0, minibatch = True )
                    y_pred = y_pred + ( w_Kelvinlets * u )

                # Compute metrics
                y_inv = y * y_std + y_mean  # Inverse transform y
                y_pred_inv = y_pred * y_std + y_mean  # Inverse transform y_pred

                log_def_mean.append( deformation_capture_mean( y_pred_inv, y_inv ) )
                log_def_max.append( deformation_capture_max( y_pred_inv, y_inv ) )
                log_def_med.append( deformation_capture_median( y_pred_inv, y_inv ) )
        
        # Aggregate metrics for testing phase
        epochMetrics[ f'{phase}_def_avg' ].append( np.mean( log_def_mean ) )
        epochMetrics[ f'{phase}_def_max' ].append( np.mean( log_def_max) )
        epochMetrics[ f'{phase}_def_median' ].append( np.mean( log_def_med ) )

        scheduler.step()

        if epoch % 10 == 0:
            print( f"Def Avg. acc [train]: {epochMetrics['train_def_avg'][-1]:.2f}" )
            print( f"Def Avg. acc [test]: {epochMetrics['test_def_avg'][-1]:.2f}" )
    
    # Save model and optimizer state
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        save_name
    )

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--save_name", type = str, required = True )
    parser.add_argument( "--device", type = str, required = True )
    
    args = parser.parse_args()
    train_test( args.save_name, args.device )
"""
