# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import sklearn.metrics
import json
from io import StringIO


fastmf_path = os.path.join('D:\\', 'FastMF_python-paper', 'FastMF_python')
if(not(fastmf_path in sys.path)):
    sys.path.insert(0,fastmf_path)
import fastmf.models.MLP_Split as MLP_Split
import fastmf.utils.NN_utils as nnu




device = 'cuda'

#Some parameters
TRAIN = True
SAVE = True


#%% Load data
base_path = ['D:\\', 'Documents\\','DATASIM\\','stage\\','data_MLP\\','dataPaperVeryBig']
task_name = 'PaperCSF'
session_id = 'PaperCSF'
type_set = 'standard'
in_normalization = 'SumToOne'
target_normalization = 'minmax'

data = MLP_Split.DataLoader(base_path, task_name, session_id, type_set, 
                                   in_normalization, target_normalization)

idx_start_train = 0
idx_end_train = 1000000
ntrain = idx_start_train - idx_end_train
ntest = 100000
nvalid = 100000
in_train = data[0][idx_start_train:idx_end_train,:]
in_valid = data[1][0:nvalid,:]
in_test = data[2][0:ntest,:]

target_train = data[3][idx_start_train:idx_end_train,:]
target_valid = data[4][0:nvalid,:]
target_test = data[5][0:ntest,:]
del data

#%% Check target data
# hists = []
# for k in range(target_train.shape[1]):
#     hist = np.histogram(target_train[:,k], bins = 60, range = [0,1] )
#     bin_middles = (hist[1][1:] + hist[1][0:-1])/2
#     bin_size = hist[1][1] - hist[1][0]
#     fig,ax = plt.subplots(2,1)
#     ax[0].plot(bin_middles, hist[0], '-og', label = 'train')
#     ax[0].legend()
    
#     hist = np.histogram(target_valid[:,k], bins = 60, range = [0,1] )
#     bin_middles = (hist[1][1:] + hist[1][0:-1])/2
#     bin_size = hist[1][1] - hist[1][0]
#     ax[1].plot(bin_middles, hist[0], '-ok', label = 'valid')
#     ax[1].legend()

#%% Load scaler and define utility function

scaler_path = os.path.join(*base_path, 'scaler', 'scaler-minmax_ses-{0}_NNLS.pickle'.format(session_id))
with open( scaler_path, 'rb') as file:
    scaler = pickle.load(file)

maxis = scaler.data_max_[:,np.newaxis]
minis = scaler.data_min_[:,np.newaxis]

def MinMaxScaler(x, minis, maxis, inverse = False):
    # if(not(np.any(x.shape == mini.shape) and np.any(x.shape == maxi.shape))):
    #     raise ValueError('Shape mismatch ! x : {0}, mini : {1}, maxi : {2}'.format(x.shape,
    #                                                                     maxi.shape,
    #                                                                     mini.shape))
    if(inverse):
        a = maxis - minis
        b = minis
    else:
        a = 1/(maxis - minis)
        b = - minis * a
    return x*a + b



#%% Define model

#Reference model (Louise Adam) without dropout // dropouts are 0.1 for each layer in the ref model
num_fasc = 2
num_atoms = in_test.shape[1]//num_fasc
num_outputs = target_test.shape[1]

p_split = 0.1#Drop out rate in the split layers
p_final = 0.1#Drop out rate in the final layers

split_architecture = [['FCL', num_atoms,num_atoms],
                      ['Activation-ReLU'],
                      ['Dropout', 0.1],
                      ['FCL', num_atoms,250],
                      ['Activation-ReLU'],
                      ]

final_architecture = [['Batchnorm', 500],
                      ['FCL', 500,250],
                      ['Activation-ReLU'],
                      ['Dropout', 0.1],
                      ['FCL', 250,250],
                      ['Activation-ReLU'],
                      ['Dropout', 0.1],
                      ['FCL', 250,100],
                      ['Activation-ReLU'],
                      ['Dropout', 0.1],
                      ['FCL', 100,num_outputs],
                      ['Activation-Sigmoid'],
                      ]



model = MLP_Split.Network(split_architecture, final_architecture)
print(model)
num_parameters = nnu.count_parameters(model)


base_save_path = 'D:\\Documents\\DATASIM\\stage\\saved_models'
folder = 'MLP_split_no_csf'
save_path = os.path.join(base_save_path, folder)
if not os.path.exists(save_path):
    # create the directory
    os.makedirs(save_path)

#%% training
if(TRAIN):
    ind_fig = 0
    num_epochs = 25
    learning_rate = 3.5e-4
    num_train_samples = in_train.shape[0]
    batch_size = 5000
    
    loss_fn = torch.nn.L1Loss(reduction = 'mean')
    
    out = MLP_Split.Train(model, batch_size, num_epochs, learning_rate, 
                  torch.from_numpy(in_train), torch.from_numpy(in_valid), 
                  torch.from_numpy(target_train), torch.from_numpy(target_valid), 
                  device = 'cuda',
                full_train_on_gpu = False,
                valid_on_gpu = True,
                bavard = 1,
                random_seed = 10, 
                loss_function = loss_fn)
    
    
    
    validation_acc = [float(x.detach().numpy()) for x in out['validation_accuracy']]
    train_losses = [float(x.detach().numpy()) for x in out['train_losses']]
    validation_losses = [float(x.detach().numpy()) for x in out['validation_losses']]
        

#%% Plot training curves
plt.figure()
plt.plot(validation_losses, label = 'validation')
plt.plot(train_losses, label = 'train')
plt.title('MSE Loss')
plt.legend()
if(SAVE):
    plt.savefig(str(ind_fig)+'.png', bbox_inches = 'tight')
    ind_fig+=1
plt.show()

plt.figure()
plt.plot([float(x) for x in validation_acc], label = 'validation MAE')
plt.legend()
if(SAVE):
    plt.savefig(str(ind_fig)+'.png', bbox_inches = 'tight')
    ind_fig+=1
plt.show()
        


#%% Prediction on test set  

model.eval() #Eval mode : affects dropout

#Evaluate on the test set
pred_test = model.cpu()(torch.from_numpy(in_test))
pred_test = pred_test.detach().cpu().numpy()


pred_valid = model.cpu()(torch.from_numpy(in_valid))
pred_valid = pred_valid.detach().cpu().numpy()

pred_train = model.cpu()(torch.from_numpy(in_train))
pred_train = pred_train.detach().cpu().numpy()

MAE_train = sklearn.metrics.mean_absolute_error(target_train,pred_train, multioutput = 'raw_values')
MAE_valid = sklearn.metrics.mean_absolute_error(target_valid,pred_valid, multioutput = 'raw_values')

MAE_test = sklearn.metrics.mean_absolute_error(target_test,pred_test, multioutput = 'raw_values')
MAE_test_avg = np.array([np.mean(MAE_test[0:num_fasc]), np.mean(MAE_test[num_fasc:])])

print('MAE Test : {0}'.format(MAE_test))
print('MAE Test Avg : {0}'.format(MAE_test_avg))

D2_MAE = sklearn.metrics.d2_absolute_error_score(target_test,pred_test, multioutput = 'raw_values')
print('D2_MAE Test : {0}'.format(D2_MAE))

Var_test = np.var(pred_test, axis = 0)
Var_GT = np.var(target_test, axis = 0)
mean_test = np.mean(pred_test, axis = 0)
mean_GT = np.mean(target_test, axis = 0)





#%% Save results

#Scale back to physical values
pred_test_ph = MinMaxScaler(pred_test, minis.T, maxis.T, inverse = True)
target_test_ph = MinMaxScaler(target_test, minis.T, maxis.T, inverse = True)
MAE_test_ph = sklearn.metrics.mean_absolute_error(target_test_ph,pred_test_ph, multioutput = 'raw_values')

s = StringIO()
print(model.Split_Layers, file=s)
inter = s.getvalue()
all_layers_split = inter.split('\n')
s.close()

s = StringIO()
print(model.Final_Layers, file=s)
inter = s.getvalue()
all_layers_final = inter.split('\n')
s.close()

if(SAVE):
    metadata = {
        'device':device,
        'base_path': base_path,
        'task_name':task_name,
        'session_id':session_id,
        'type_set':type_set,
        'in_normalization':in_normalization,
        'target_normalization':target_normalization,
        'num_atoms': num_atoms,
        'num_outputs':num_outputs,
        'num_fasc':num_fasc,
        'split_architecture':split_architecture,
        'final_architecture':final_architecture,
        'layers split':all_layers_split,
        'layers final':all_layers_final,
        'number of parameters': num_parameters,
        'num_epochs':num_epochs,
        'learning_rate':learning_rate,
        'num_train_samples':num_train_samples,
        'batch_size':batch_size,
        'MAE_test':MAE_test.tolist(),
        'MAE_test_ph':MAE_test_ph.tolist(),
        'MAE_test_avg':MAE_test_avg.tolist(),
        'D2_MAE':D2_MAE.tolist(),
        'variance of predictions on test set':Var_test.tolist(),
        'mean of predictions on test set':mean_test.tolist(),
        'variance of ground truth values':Var_GT.tolist(),
        'mean of ground truth values':mean_GT.tolist(),
        'validation_acc':validation_acc,
        'validation_losses':validation_losses,
        'train_losses':train_losses,


        }
    # open the file for writing
    with open(os.path.join(save_path,'metadata.json'), 'w') as f:
        # write the data to the file
        json.dump(metadata, f, indent = 0)
        
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict.pt'))