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
import fastmf.models.MLP_FullyLearned as MLP_Full
import fastmf.utils.NN_utils as nnu

device = 'cuda'

#Some parameters
TRAIN = True
SAVE = True


#%% Load data
base_path = ['D:\\', 'Documents\\','DATASIM\\','stage\\','data_MLP\\','dataPaperVeryBig']
task_name = 'PaperCSF'
session_id = 'BigPaperCSF'
type_set = 'standard'
in_normalization = 'None'
target_normalization = 'minmax'

in_train_cubic, in_valid_cubic, in_test_cubic, target_train, target_valid, target_test = MLP_Full.DataLoader(base_path, task_name, session_id, type_set,
                                   in_normalization, target_normalization)

ntrain = in_train_cubic.shape[0]#2000000
nvalid = in_valid_cubic.shape[0]#100000
ntest = in_test_cubic.shape[0]#100000

in_train = in_train_cubic[0:ntrain,:].reshape(ntrain, in_train_cubic.shape[1]*in_train_cubic.shape[2])
del in_train_cubic
in_valid = in_valid_cubic[0:nvalid,:].reshape(nvalid, in_valid_cubic.shape[1]*in_valid_cubic.shape[2])
del in_valid_cubic
in_test = in_test_cubic[0:ntest,:].reshape(ntest, in_test_cubic.shape[1]*in_test_cubic.shape[2])
del in_test_cubic

target_train = target_train[0:ntrain,:]
target_valid = target_valid[0:nvalid,:]
target_test = target_test[0:ntest,:]



#%% Load scaler and define utility function

scaler_path = os.path.join(*base_path, 'scaler', 'scaler-minmax_ses-{0}_SH.pickle'.format(session_id))
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
num_fasc = 2
num_atoms = in_test.shape[1]//num_fasc
num_outputs = target_test.shape[1]
num_inputs = in_train.shape[1]

p_split = 0.1#Drop out rate in the split layers
p_final = 0.1#Drop out rate in the final layers



architecture = [['FCL', num_inputs,1000],
                ['Activation-ReLU'],
                ['Dropout', 0.1],
                ['FCL', 1000,1000],
                ['Activation-ReLU'],
                ['Dropout', 0.1],
                ['FCL', 1000,1000],
                ['Activation-ReLU'],
                ['Dropout', 0.1],
                ['FCL', 1000,num_outputs],
                ['Activation-Sigmoid'],
                ]



model = MLP_Full.Network(architecture)
num_parameters = nnu.count_parameters(model)
print('Number of parameters : ', num_parameters)

base_save_path = 'D:\\Documents\\DATASIM\\stage\\saved_models'
folder = 'MLP_FullyLearned_csf_4'
save_path = os.path.join(base_save_path, folder)
if not os.path.exists(save_path):
    # create the directory
    os.makedirs(save_path)

#%% training
if(TRAIN):
    ind_fig = 0
    num_epochs = 70
    learning_rate = 2e-4
    num_train_samples = in_train.shape[0]
    batch_size = 12000
    metric_function = lambda x,y:MLP_Full.D2score(x,y)
    out = MLP_Full.Train(model, batch_size, num_epochs, learning_rate, 
                  torch.from_numpy(in_train), torch.from_numpy(in_valid), 
                  torch.from_numpy(target_train), torch.from_numpy(target_valid), 
                  device = 'cuda',
                full_train_on_gpu = False,
                valid_on_gpu = True,
                bavard = 1,
                random_seed = 10, 
                loss_function = torch.nn.L1Loss(),
                metric_function = metric_function)
    
    
    try:
        validation_acc = [float(x.detach().numpy()) for x in out['validation_accuracy']]
    except AttributeError:
        validation_acc = [float(x) for x in out['validation_accuracy']]
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

MAE_test = sklearn.metrics.mean_absolute_error(target_test,pred_test, multioutput = 'raw_values')
MAE_test_avg = np.array([np.mean(MAE_test[0:num_fasc]), np.mean(MAE_test[num_fasc:-1])])

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
print(model.Layers, file=s)
inter = s.getvalue()
all_layers = inter.split('\n')
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
        'architecture':architecture,
        'layers':all_layers,
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