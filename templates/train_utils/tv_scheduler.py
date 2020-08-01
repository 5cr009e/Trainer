from .logger import Logger
import numpy as np
from tqdm import tqdm
from .schedulers.scheduler import get_scheduler
from .optim.optimizers import get_optimizer

class TrainValScheduler:
    def __init__(self, network, data_loaders,
                 lr=1e-3, num_epoch=5000, batch_size=128,
                 using_step_lr=True, step_size=50, gamma=0.9,
                 loss_type="l1_loss", weight_save_epochs=50,
                 logging_path="log/", optimizer_type="Adadelta"):
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.using_step_lr = using_step_lr
        self.step_size = step_size
        self.gamma = gamma
        self.loss_type = loss_type
        self.weight_save_epochs = weight_save_epochs
        self.logging_path = logging_path
        self.optimizer_type = optimizer_type

        self.network = network
        self.train_loader, self.test_loader = data_loaders        
        self.optimizer = get_optimizer(optimizer_type, self.network.network.parameters(), {"lr": lr})

        if using_step_lr:
            self.scheduler = get_scheduler()

        self.loss_dict = {}

    def train_epoch(self, training_iteration):
        # training
        train_loss_list = []
        for data, label in self.train_loader:
            loss = self.network.train({"data": data, "label": label}, self.optimizer)
            train_loss_list.append(loss.item())
            training_iteration += 1
        if self.using_step_lr:
            self.scheduler.step()
        self.loss_dict.update(self.network.collate_loss(train_loss_list, 'train_loss'))

    def validate_epoch(self, training_iteration):
        # validating
        for data, label in self.test_loader:
            eval_loss_list = []
            loss = self.network.validate({"data": data, "label": label})
            eval_loss_list.append(loss.item())
        self.loss_dict.update(self.network.collate_loss(eval_loss_list, 'eval_loss'))

    def train_val(self, train_flag=True, validate_flag=True):
        training_iteration = 0
        with tqdm(range(self.num_epoch+1), total=self.num_epoch+1) as pbar:
            for i in range(self.num_epoch+1):
                if train_flag: self.train_epoch(training_iteration)
                if validate_flag: self.validate_epoch(training_iteration) 
                pbar.set_postfix()
                if i % self.weight_save_epochs == 0:
                    self.network.save(i)
                pbar.update(1)
