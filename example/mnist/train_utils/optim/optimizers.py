import torch

def get_optimizer(optimizer_type, model_params, params_dict):
    return eval("torch.optim.{}".format(optimizer_type))(model_params, lr=params_dict['lr']) 
