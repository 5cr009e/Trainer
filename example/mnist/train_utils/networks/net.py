from abc import ABC, abstractmethod

class NetworkWrapper(ABC):
    def __init__(self, network, loss_type, save_path, device, tag):
        self.network = network
        self.device = device
        self.loss_type = loss_type
        self.network.to(device)

        self.save_path = save_path
        self.tag = tag
    def preprocess(self, data_dict):
        data = data_dict['data']
        label = data_dict['label']
        data, label = data.to(self.device), label.to(self.device)
        return data, label

    @abstractmethod
    def train(self, data_dict, optimizer):
        pass

    @abstractmethod
    def train(self, data_dict, optimizer):
        pass

    @abstractmethod
    def get_loss(self, label, output):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def collate_loss(loss_val, train_val_type):
        pass
    
    @abstractmethod
    def log_dict(train_loss, eval_loss):
        pass
    # return {'train_loss': '{0:1.5f}'.format(train_loss),
                #  'eval_loss': '{0:1.5f}'.format(eval_loss)}
