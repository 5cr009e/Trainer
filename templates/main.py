
from dataset.dataset import get_loader
from train_utils.networks.cnn import CNNWrapper

import click
from train_utils.tv_scheduler import TrainValScheduler


@click.command()
@click.option('--lr', default=1e-3, help='learning_rate')
@click.option('--num_epoch', default=500, help='num_epoch')
@click.option('--batch_size', default=128, help='batch_size')
@click.option('--loss_type', default='nll_loss')
@click.option('--network_tag', default='default')
def main(lr, num_epoch, batch_size, 
    loss_type, network_tag):

    net = CNNWrapper(loss_type=loss_type, save_path="./output", device="cuda", tag=network_tag)
    data_loaders = get_loader(batch_size=batch_size)
    
    tvs = TrainValScheduler(network=net, data_loaders=data_loaders,
                            lr=1e-3, num_epoch=1, batch_size=200,
                            using_step_lr=False, step_size=50, gamma=0.9,
                            weight_save_epochs=50,
                            logging_path="log/", optimizer_type="Adadelta")
    tvs.train_val()
if __name__ == '__main__':
    main()