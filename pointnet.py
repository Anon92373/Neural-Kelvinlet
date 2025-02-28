import torch
from torch import nn
from torch.nn import ReLU, Linear
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor,self).__init__()
        self.fc1 = nn.Linear(1024 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1).to(x.device)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = False, feature_transform = True):
        super(PointNetfeat, self).__init__()
        #self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(10, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.regressor=Regressor()

    def forward(self, x):
        x=x.permute(0,2,1)
        n_pts = x.size()[2]
        # trans = self.stn(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return self.regressor(torch.cat([x, pointfeat], 1).transpose(-2,-1)),trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss
        
        
        
class PointNetV2(nn.Module):
    def __init__(self, global_feat = False, feature_transform = True):
        super(PointNetV2, self).__init__()
        #self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(4, 128, 1)
        self.fstn1 = STNkd(k=128)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.fstn2 = STNkd(k=256)
        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.fstn = STNkd(k=128)
        self.regressor=Regressor()

    def forward(self, x):
        x=x.permute(0,2,1)
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        
        trans_feat = self.fstn1(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)

        pointfeat = x
        
        x = F.relu(self.bn2(self.conv2(x)))
        trans_feat = self.fstn2(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)
        
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return self.regressor(torch.cat([x, pointfeat], 1).transpose(-2,-1)), None, trans_feat
        
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss