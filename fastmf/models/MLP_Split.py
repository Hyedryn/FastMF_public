# -*- coding: utf-8 -*-
import torch
import time
from tqdm import tqdm
import torch.optim as optim
import fastmf.utils.NN_utils as nnu
import os
import numpy as np


#%% Utility d2 mae score
def D2score(pred,target):
    gt_medians = torch.median(target, axis = 0)[0]
    num = torch.mean(torch.abs(pred - target))
    den = torch.mean(torch.abs(target - gt_medians.unsqueeze(0)))
    return 1 - num/den

#%% Split MLP class
class Network(torch.nn.Module):
    def __init__(self, split_architecture, final_architecture):
        super(Network, self).__init__()
        self.split_architecture = split_architecture
        self.final_architecture = final_architecture
        self.Split_Layers = torch.nn.ModuleList([torch.nn.Sequential(), torch.nn.Sequential()])
        self.Final_Layers = torch.nn.Sequential()
        
        for i,module_data in enumerate(split_architecture): #Split layers
            name = module_data[0].split('-')
            if(name[0]=='FCL'):
                
                self.Split_Layers[0].add_module('Layer_0_{0}'.format(i),
                                       torch.nn.Linear(module_data[1],
                                                       module_data[2])                                   
                                   )
                self.Split_Layers[1].add_module('Layer_1_{0}'.format(i),
                                       torch.nn.Linear(module_data[1],
                                                       module_data[2])                                   
                                   )
            elif(name[0] == 'Dropout'):
                self.Split_Layers[0].add_module('Dropout_0_{0}'.format(i),
                                   torch.nn.Dropout(p=module_data[1]))
                self.Split_Layers[1].add_module('Dropout_1_{0}'.format(i),
                                   torch.nn.Dropout(p=module_data[1]))
            elif(name[0] == 'Batchnorm'):
                self.Split_Layers[0].add_module('Batchnorm_0_{0}'.format(i),
                                   torch.nn.Batchnorm1d(module_data[1]))
                self.Split_Layers[1].add_module('Batchnorm_1_{0}'.format(i),
                                   torch.nn.Batchnorm1d(module_data[1]))
            elif(name[0] == 'Activation'):
                nnu.add_activation(self.Split_Layers[0], name[1], str_name = 'Activation_0_'+str(i))
                nnu.add_activation(self.Split_Layers[1], name[1], str_name = 'Activation_1_'+str(i))
                
            for i,module_data in enumerate(final_architecture): #Final Layers
                name = module_data[0].split('-')
                if(name[0]=='FCL'):
                    
                    self.Final_Layers.add_module('Layer_0_{0}'.format(i),
                                           torch.nn.Linear(module_data[1],
                                                           module_data[2])                                   
                                       )
                               
                                       
                elif(name[0] == 'Dropout'):
                    self.Final_Layers.add_module('Dropout_0_{0}'.format(i),
                                       torch.nn.Dropout(p=module_data[1]))

                elif(name[0] == 'Activation'):
                    nnu.add_activation(self.Final_Layers, name[1], str_name = 'Activation_0_'+str(i))

    def forward(self, x):
        num_atoms = self.split_architecture[0][1] # ['FCL', num_atoms, ?????]
        x_fasc1 = x[:,0:num_atoms]
        x_fasc2 = x[:,num_atoms:2*num_atoms]
        if(x.shape[1]==2*num_atoms+1):
            x_csf = x[:,-1:]
        for i in range(len(self.Split_Layers[0])):
             x_fasc1 = self.Split_Layers[0][i](x_fasc1)
             x_fasc2 = self.Split_Layers[0][i](x_fasc2)
        x = torch.concatenate((x_fasc1,x_fasc2), dim = 1)
        for i,module in enumerate(self.Final_Layers):
            x = module(x)
        
        return x

#%% Data loading function
def DataLoader(base_path, 
               task_name, session_id, type_set,
               input_normalization,
               target_normalization, orientation_estimate):
    db_in = os.path.join(base_path, 'formatter', 'type-standard', 'NNLS')
    base = "type-{0}_task-{1}_ses-{2}_orientation-{3}_normalization".format(type_set, task_name, session_id, orientation_estimate)

    train_input_name = base + '-{0}_set-training_NNLS.npy'.format(input_normalization)
    validation_input_name = base + '-{0}_set-validation_NNLS.npy'.format(input_normalization)
    test_input_name = base + '-{0}_set-testing_NNLS.npy'.format(input_normalization)

    train_input_path = os.path.join(db_in,train_input_name)
    validation_input_path = os.path.join(db_in,validation_input_name)
    test_input_path = os.path.join(db_in,test_input_name)


    base = "type-{0}_task-{1}_ses-{2}_orientation-{3}_scaler".format(type_set, task_name, session_id, orientation_estimate)
    train_target_name = base + '-{0}_set-training_target.npy'.format(target_normalization)
    validation_target_name = base + '-{0}_set-validation_target.npy'.format(target_normalization)
    test_target_name = base + '-{0}_set-testing_target.npy'.format(target_normalization)

    train_target_path = os.path.join(db_in,train_target_name)
    validation_target_path = os.path.join(db_in,validation_target_name)
    test_target_path = os.path.join(db_in,test_target_name)


    with open(train_input_path, 'rb') as file:
        in_train = np.load(file).astype('float32')
        
    with open(validation_input_path, 'rb') as file:
        in_valid = np.load(file).astype('float32')
        
    with open(test_input_path, 'rb') as file:
        in_test = np.load(file).astype('float32')
       
    with open(train_target_path, 'rb') as file:
        target_train = np.load(file).astype('float32')

    with open(validation_target_path, 'rb') as file:
        target_valid = np.load(file).astype('float32')
        
    with open(test_target_path, 'rb') as file:
        target_test = np.load(file).astype('float32')

    
    
    return in_train,in_valid,in_test,  target_train,target_valid,target_test



#%% Training function
#%% Training function

def Train(model, batch_size, num_epochs, learning_rate, 
              in_train, in_valid, 
              target_train,target_valid, 
              device = 'cpu',
            full_train_on_gpu = False,
            valid_on_gpu = False,
            bavard = 0,
            random_seed = 10, 
            loss_function = None,
            metric_function = None):
        
        max_grad = 0 #Maximum gradient values (over all parameters) : to monitor training
        torch.manual_seed(random_seed)
        num_train_samples = in_train.shape[0] #Number of samples in the training set

        get_slice = lambda i, size: range(i * size, (i + 1) * size) #Will be used to get the different batches
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if(loss_function is None):
            print('WARNING : Using default (L1Loss) loss function')
            loss_function = torch.nn.L1Loss(reduction = 'mean')
        #For 'accuracy' evaluation
        #Should be different than loss_fn ideally, 
        #and can be used to quickly determine the comparative performances of different loss functions
        if(metric_function is None):
            metric_function = torch.nn.L1Loss(reduction='mean') 
            
        if(device == 'cuda'):
            model = model.to(device)
            
        
        print('----------------------- Training --------------------------', flush=True)
        start = time.time() 
        
        num_batches_train = num_train_samples // batch_size #Number of training batches
        train_losses = [] #Will store the loss on the training set for each epoch
        val_losses = [] #Will store the loss on the validation set for each epoch
        val_acc = [] #Will store the 'accuracy' on the validation set for each epoch
        
        if(device == 'cuda'):
            torch.cuda.reset_peak_memory_stats(device)
            if(full_train_on_gpu):
                in_train = in_train.detach().to(device)
                target_train = target_train.detach().to(device)
        for epoch in tqdm(range(num_epochs)):
            #print('\n************ Epoch {0} ************'.format(epoch))
            if(bavard > 0):
                print('memory :', torch.cuda.memory_allocated()/1024/1024)
            for i in range(num_batches_train):
                optimizer.zero_grad() #Reset gradients
                batch = get_slice(i,batch_size) #Create indexes of current batch
                
                if(full_train_on_gpu or device == 'cpu'):
                    output = model.forward(in_train[batch, :]) #Forward pass
                    batch_loss = loss_function(output, target_train[batch, :])#, batch)#Calculate loss
                else:
                    output = model.forward(in_train[batch, :].detach().to(device)) #Forward pass
                    batch_loss = loss_function(output, target_train[batch, :].detach().to(device)) #Calculate loss

                batch_loss.backward() #Calculate gradients
                optimizer.step() #Update based on gradients
                param = [x for x in model.parameters()]
                for p in param:
                    if(torch.any(torch.isnan(p))):
                        print('Nan value encountered in model parameters')
                
                #Get maximum gradient values to monitor training 
                if(bavard >5):
                    full_grads = [x.grad for x in model.parameters()]
                    maxi = 0
                    min_grad = 1e12
                    for grad in full_grads:
                        if(not(grad is None)):
                            maxi = torch.max(torch.abs(grad))
                            mini = torch.min(torch.abs(grad))
                            max_grad = max(max_grad,maxi)
                            min_grad = min(min_grad,mini)
                    print('Epoch : {0} / Batch : {1} / Max grad so far : {2}'.format(epoch,i,max_grad))
                    print('Epoch : {0} / Batch : {1} / Min grad so far : {2}'.format(epoch,i,min_grad))
                
                del output
                del batch

                if(i<num_batches_train-1):
                    del batch_loss  #Keep the last one for each epoch
                if(device == 'cuda'):
                    torch.cuda.empty_cache()
         
            if(device == 'cuda'):
                batch_loss_cpu = batch_loss.to('cpu') #To avoid memory leak in gpu
                train_losses.append(batch_loss_cpu.detach()) #Store loss of the last batch
            else:
                train_losses.append(batch_loss.detach())
            if(bavard>0):
                print('Train loss at end of the epoch : ', batch_loss )    
            del batch_loss
            if(device == 'cuda'):
                del batch_loss_cpu
                torch.cuda.empty_cache()
            #Evaluate on validation set
            if(device == 'cuda'):
                if(valid_on_gpu):
                    if(target_valid.shape[0]<10000):
                        validation_output = model.forward(in_valid.detach().to(device),
                                                          ) #Prediction on validation set
                    else:
                        validation_output = torch.zeros(target_valid.shape, device = 'cuda', dtype = torch.float32)
                        idx_start = 0
                        while(idx_start<validation_output.shape[0]):
                            idx_end = min(validation_output.shape[0], idx_start + 10000)
                            validation_output[idx_start:idx_end,:] = model.forward(in_valid[idx_start:idx_end].detach().to(device))
                            idx_start = idx_end
                        
                    batch_loss_valid = loss_function(validation_output, target_valid.detach().to(device),)
                                                     #range(num_train_samples, num_train_samples + num_valid_samples)) #Calculate loss on validation set
                    batch_loss_valid_cpu = batch_loss_valid.detach().to('cpu') #To avoid memory leak in gpu
                    batch_acc_valid = metric_function(validation_output, target_valid.to(validation_output.device))
                    batch_acc_valid_cpu = batch_acc_valid.detach().to('cpu')
                    val_losses.append(batch_loss_valid_cpu) #Store loss of the validation set
                    val_acc.append(batch_acc_valid_cpu) #Store MAE on the validation set
                else:
                    validation_output = model.to('cpu').forward(in_valid) #Prediction on validation set
                    batch_loss_valid = loss_function(validation_output, target_valid, ) #Calculate loss on validation set
                    batch_acc_valid = metric_function(validation_output.detach(), target_valid.detach())
                    val_losses.append(batch_loss_valid) #Store loss of the validation set
                    val_acc.append(batch_acc_valid) #Store MAE on the validation set
                    
                    model.to(device)

            else:
                validation_output = model.forward(in_valid, )#Prediction on validation set
                batch_loss_valid = loss_function(validation_output, target_valid) #Calculate loss on validation set
                batch_acc_valid = metric_function(validation_output, target_valid)
                val_losses.append(batch_loss_valid.detach()) #Store loss of the validation set
                val_acc.append(batch_acc_valid.detach()) #Store MAE on the validation set
                   
            if(bavard>0):
                print('Validation loss at end of the epoch : ', batch_loss_valid)
            if(bavard >1):
                print('memory before deletions :', torch.cuda.memory_allocated()/1024/1024)    
            
            
            del batch_loss_valid
            del validation_output
            del batch_acc_valid
            if(device == 'cuda'):
                if(bavard>0):
                    print('\n\nMax GPU memory used during epoch : ', torch.cuda.max_memory_allocated(device)/1024/1024)
                if(valid_on_gpu):
                    del batch_loss_valid_cpu
                    del batch_acc_valid_cpu
                torch.cuda.empty_cache()
            if(bavard>1):
                print('memory after deletions :', torch.cuda.memory_allocated()/1024/1024)    
                
        
        end = time.time()
        total_time = end-start
        out = {"train_losses": train_losses,
               "validation_losses": val_losses,
               "validation_accuracy" : val_acc,
               "model": model,
               "total_time": total_time,
               "optimizer": optimizer,
               }
        if(device == 'cuda'):
            torch.cuda.empty_cache()
        return out