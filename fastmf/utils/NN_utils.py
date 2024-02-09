# -*- coding: utf-8 -*-
import torch


#%% Utility function : add_activation
def add_activation(layers, activation_name, str_name = 'Activation'):
    if(activation_name == 'LeakyReLU'):
        layers.add_module(str_name,torch.nn.LeakyReLU())  
    elif(activation_name == 'ReLU'):
        layers.add_module(str_name,torch.nn.ReLU())  
    elif(activation_name == 'SELU'):
        layers.add_module(str_name,torch.nn.SELU())  
    elif(activation_name == 'ELU'):
        layers.add_module(str_name,torch.nn.ELU())  
    elif(activation_name == 'SiLU'):
        layers.add_module(str_name,torch.nn.SiLU())  
    elif(activation_name == 'Mish'):
        layers.add_module(str_name,torch.nn.Mish())  
    elif(activation_name == 'Softsign'):
        layers.add_module(str_name,torch.nn.Softsign())   
    elif(activation_name == 'Tanhshrink'):
        layers.add_module(str_name,torch.nn.Tanhshrink())  
    elif(activation_name == 'Tanh'):
        layers.add_module(str_name,torch.nn.Tanh())  
    elif(activation_name == 'ELU'):
        layers.add_module(str_name,torch.nn.ELU())  
    elif(activation_name == 'Sigmoid'):
        layers.add_module(str_name,torch.nn.Sigmoid())  
    else:
        print('Warning : Putting ReLU (default) as activation')
        layers.add_module(str_name,torch.nn.ReLU()) 
#%% utility function to get the number of trainable parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)