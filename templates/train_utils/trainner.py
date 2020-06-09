from .logger import Logger
import numpy as np
from tqdm import tqdm
from .schedulers.scheduler import get_scheduler
from .optim.optimizers import get_optimizer


def train_network(network, data_loaders, network_name,
                  lr=1e-3, epochs=5000, batch=128,
                  using_step_lr=True, step_size=50, gamma=0.9,
                  loss_type="l1_loss", weight_save_epochs=50,
                  logging_path="log/", optimizer_str="Adam"):
    train_loader, test_loader = data_loaders
    optimizer = get_optimizer(optimizer_str, {"lr": lr})
    if using_step_lr:
        scheduler = get_scheduler()

    logger = Logger(logging_path)
    training_iteration = 0
    with tqdm(range(epochs+1), total=epochs+1) as pbar:
        for i in range(epochs+1):
            for data, label in train_loader:
                loss = network.train({"data": data, "label": label}, optimizer)
                logger.train_step(loss, training_iteration)
                training_iteration += 1
            if using_step_lr:
                scheduler.step(i)
            network.eval()
            for data, label in test_loader:
                eval_loss = []
                loss = network.eval({"data": data, "label": label})
                eval_loss.append(loss.item())
            logger.eval_step(np.mean(eval_loss), training_iteration)

            pbar.set_postfix({'eval_loss': '{0:1.5f}'
                             .format(np.mean(eval_loss))})
            if i % weight_save_epochs == 0:
                network.save(i)
            pbar.update(1)
