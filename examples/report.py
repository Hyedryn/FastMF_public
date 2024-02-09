import os
import pickle
import sys
from sklearn.metrics import d2_absolute_error_score
import numpy as np
import torch

fastmf_path = os.path.join('D:\\','FastMF_python-paper', 'FastMF_python')
if(not(fastmf_path in sys.path)):
    sys.path.insert(0,fastmf_path)
import fastmf.reports.evaluator as evaluator

include_csf = True

base_path = ['D:\\', 'Documents\\','DATASIM\\','stage\\','data_MLP\\','dataTestPaper3']

model_base_path = ['D:\\', 'Documents', 'DATASIM', 'stage', 'saved_models']

hybrid_model_path = os.path.join(*model_base_path, 
                          'MLP_split_paper_csf')

full_model_path = os.path.join(*model_base_path, 
                          'MLP_FullyLearned_csf_4')

task_name = "PaperCSF"
run_id = "0"
session_id = "PaperCSF"


metadata_Hybrid = os.path.join(hybrid_model_path, 'metadata.json')
model_state_dict_Hybrid = os.path.join(hybrid_model_path, 'model_state_dict.pt')

metadata_Full = os.path.join(full_model_path, 'metadata.json')
model_state_dict_Full = os.path.join(full_model_path, 'model_state_dict.pt')


#%%
evaluation = evaluator.Evaluator(os.path.join(*base_path), 
                                 task_name, session_id, run_id, 
                           metadata_Hybrid, model_state_dict_Hybrid, 
                           metadata_Full, model_state_dict_Full,
                           scaling_fn = 'MinMax')


#%% Compute basic metrics

d2_MF_MSMT = d2_absolute_error_score(evaluation.target_NNLS[:,:], 
                                     evaluation.scaling_fn_hybrid(evaluation.MF_MSMTCSD_output,
                                    evaluation.minis_hybrid,
                                    evaluation.maxis_hybrid,
                                    inverse = False
                                    ),
                                     multioutput = 'raw_values')

d2_MF_GT = d2_absolute_error_score(evaluation.target_NNLS[:,:], 
                                     evaluation.scaling_fn_hybrid(evaluation.MF_GROUNDTRUTH_output,
                                    evaluation.minis_hybrid,
                                    evaluation.maxis_hybrid,
                                    inverse = False
                                    ),
                                     multioutput = 'raw_values')

d2_Hybrid = d2_absolute_error_score(evaluation.target_NNLS, 
                                    evaluation.pred_hybrid, 
                                    multioutput = 'raw_values')

idx_prop_full = [0,4,5,6,10,11,12]
d2_Full = d2_absolute_error_score(evaluation.target_SH[:,idx_prop_full], 
                                  evaluation.pred_full[:,idx_prop_full], 
                                  multioutput = 'raw_values')


print('D2 MF MSMT : ', d2_MF_MSMT)
print('D2 MF GT :', d2_MF_GT)
print('D2 Hybrid : ', d2_Hybrid)
print('D2 Full : ', d2_Full)

#%%
evaluation._Evaluator__plot_MAEbyAngularError()

#%%
evaluation._Evaluator__plot_GTvsPrediction()

#%%
evaluation._Evaluator__plot_Residuals()

#%%
evaluation._Evaluator__plot_Distributions()

#%%
abs_dots_MF, abs_dots_SH = evaluation._Evaluator__plot_AngularError()

angles_MF = np.arccos(abs_dots_MF)*180/np.pi
med = np.median(angles_MF)
avg = np.mean(angles_MF)
q1 = 0.25
qtile1 = np.quantile(angles_MF, q1)
q2 = 0.75
qtile2 = np.quantile(angles_MF,q2)
q3 = 0.95
qtile3 = np.quantile(angles_MF,q3)
print('MSMT Median angular error : ', med)
print('MSMT Average angular error : ', avg)
print('MSMT Quantile \ q = {0} : {1}'.format(q1,qtile1))
print('MSMT Quantile \ q = {0} : {1}'.format(q2,qtile2))
print('MSMT Quantile \ q = {0} : {1}'.format(q3,qtile3))


angles_SH = np.arccos(abs_dots_SH)*180/np.pi
med = np.median(angles_SH)
avg = np.mean(angles_SH)
q1 = 0.25
qtile1 = np.quantile(angles_SH, q1)
q2 = 0.75
qtile2 = np.quantile(angles_SH,q2)
q3 = 0.95
qtile3 = np.quantile(angles_SH,q3)
print('SH Median angular error : ', med)
print('SH Average angular error : ', avg)
print('SH Quantile \ q = {0} : {1}'.format(q1,qtile1))
print('SH Quantile \ q = {0} : {1}'.format(q2,qtile2))
print('SH Quantile \ q = {0} : {1}'.format(q3,qtile3))


#%%
evaluation._Evaluator__plot_MAE_By_Nu()

#%%
evaluation._Evaluator__plot_D2_by_nu()

#%%
evaluation._Evaluator__assess_nuCSF()

#%%
