# optimizer.py
# optimizers to train neural networks
import torch

def optimizer(model, lr, kwargs):
    if kwargs == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif kwargs == "rmsprop":
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif kwargs == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif kwargs == "adamw":
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr)
    elif kwargs == "lbfgs":
        return torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr)
    else:
        return NotImplementedError
    
def loss_fun(loss_name):
    if loss_name == 'l2':
        return torch.nn.MSELoss()
    elif loss_name == 'l1':
        return torch.nn.L1Loss()
    elif loss_name == 'poisson':
        return torch.nn.PoissonNLLLoss()
    elif loss_name == 'huber':
        return torch.nn.HuberLoss()
    else:
        return NotImplementedError