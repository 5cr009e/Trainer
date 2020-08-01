from .net import NetworkWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        output = F.log_softmax(self.fc2(x), dim=1)
        return output

class CNNWrapper(NetworkWrapper):
    def __init__(self, loss_type, save_path, device, tag):
        super(CNNWrapper, self).__init__(CNN(), loss_type, save_path, device, tag)
    
    def get_loss(self, label, output):
        return eval("torch.nn.functional." + self.loss_type)(output, label)

    def train(self, data_dict, optimizer):
        self.network.train()
        data, label = self.preprocess(data_dict)
        optimizer.zero_grad()
        output = self.network(data)
        return self.get_loss(label, output)

    def validate(self, data_dict):
        self.network.eval()
        data, label = self.preprocess(data_dict)
        output = self.network(data)
        return self.get_loss(label, output)

    @staticmethod
    def collate_loss(loss_val, loss_type):
        return {loss_type: np.mean(loss_val)}

    @staticmethod
    def log_dict(train_loss, eval_loss):
        return {'train_loss': '{0:1.5f}'.format(train_loss),
                'eval_loss': '{0:1.5f}'.format(eval_loss)}
    
    def save(self, ep):
        ep_save_path = "{}/{}/".format(self.save_path, self.tag)
        Path("ep_save_path").mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(ep_save_path, "ep{}.pth".format(ep)))
