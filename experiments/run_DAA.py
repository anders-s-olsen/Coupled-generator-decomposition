import sys
import os.path
import torch
import numpy as np
from TMMSAA import TMMSAA, TMMSAA_trainer
from ica import ica1
#import matplotlib.pyplot as plt
#from TMMSAA.TMMSAA import SSE
torch.set_num_threads(16)

modality_names = ["EEG", "MEG"]

Xtrain_mmmsmc = {} # split by modality, subject, condition
Xtrain_mmms = {} # split by modality, subject
Xtrain_mm = {} # split by modality
for m in modality_names:
    Xtrain_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob0.pt")
    Xtrain_mmms[m] = torch.cat((Xtrain_mmmsmc[m][:, 0], Xtrain_mmmsmc[m][:, 1],Xtrain_mmmsmc[m][:, 2],),dim=-1)
    Xtrain_mm[m] = torch.reshape(Xtrain_mmms[m],(16*Xtrain_mmms[m].shape[-2],540))
Xtrain_group_daa = torch.cat((Xtrain_mm['EEG'],Xtrain_mm['MEG']),dim=-2)
Xtrain = {'group_daa':Xtrain_group_daa,'mm_daa':Xtrain_mm,'mmms_daa':Xtrain_mmms}

Xtest_mmmsmc = {} # split by modality, subject, condition
Xtest_mmms = {} # split by modality, subject
Xtest_mm = {} # split by modality
for m in modality_names:
    Xtest_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_frob1.pt")
    Xtest_mmms[m] = torch.cat((Xtest_mmmsmc[m][:, 0], Xtest_mmmsmc[m][:, 1],Xtest_mmmsmc[m][:, 2],),dim=-1)
    Xtest_mm[m] = torch.reshape(Xtest_mmms[m],(16*Xtest_mmms[m].shape[-2],540))
Xtest_group_daa = torch.cat((Xtest_mm['EEG'],Xtest_mm['MEG']),dim=-2)
Xtest = {'group_daa':Xtest_group_daa,'mm_daa':Xtest_mm,'mmms_daa':Xtest_mmms}

Xtraintilde_mmmsmc = {} # split by modality, subject, condition
Xtraintilde_mmms = {} # split by modality, subject
Xtraintilde_mm = {} # split by modality
for m in modality_names:
    Xtraintilde_mmmsmc[m] = torch.load("data/concatenatedData/X_" + m + "_FT_l20.pt")
    Xtraintilde_mmms[m] = torch.cat((Xtraintilde_mmmsmc[m][:, 0], Xtraintilde_mmmsmc[m][:, 1],Xtraintilde_mmmsmc[m][:, 2],),dim=-1)
    Xtraintilde_mm[m] = torch.reshape(Xtraintilde_mmms[m],(16*Xtraintilde_mmms[m].shape[-2],540))
Xtraintilde_group_daa = torch.cat((Xtraintilde_mm['EEG'],Xtraintilde_mm['MEG']),dim=-2)
Xtraintilde = {'group_daa':Xtraintilde_group_daa,'mm_daa':Xtraintilde_mm,'mmms_daa':Xtraintilde_mmms}

daa_modeltypes=['group_daa','mm_daa','mmms_daa']
num_modalities=[1,2,2]
times = torch.load("data/MEEGtimes.pt")
dims = {'group_daa':Xtrain_group_daa.shape,'mm_daa':Xtrain_mm["EEG"].shape,'mmms_daa':Xtrain_mmms["EEG"].shape}
#C_idx = torch.hstack((torch.zeros(20, dtype=torch.bool), torch.ones(160, dtype=torch.bool)))

l1_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
l2_vals = torch.hstack((torch.tensor(0),torch.logspace(-5,2,8)))
#l2_vals = l2_vals[2:]

num_iter_outer = 5
num_iter_inner = 20

for K in range(5,21):

    
    for m,modeltype in enumerate(daa_modeltypes):
        if m==0 or m==1:
            continue
        if os.path.isfile("data/DAA_results/train_loss_"+modeltype+"_K="+str(K)+'.txt'):
            continue
        daa_train_loss = np.zeros((num_iter_outer,num_iter_inner))
        daa_test_loss = np.zeros((num_iter_outer,num_iter_inner))
        for outer in range(num_iter_outer):
            for inner in range(num_iter_inner):
                print('K='+str(K)+', '+modeltype+' '+str(outer)+'_'+str(inner))
                model = TMMSAA.TMMSAA(dimensions=dims[modeltype],num_comp=K,num_modalities=num_modalities[m],model='DAA')
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                loss = TMMSAA_trainer.Optimizationloop(model=model,X=Xtrain[modeltype],Xtilde=Xtraintilde[modeltype],optimizer=optimizer,max_iter=10000,tol=1e-4)

                daa_test_loss[outer,inner] = model.eval_model(Xtrain=None,Xtraintilde=Xtrain[modeltype],Xtest=Xtest[modeltype])
                daa_train_loss[outer,inner] = model.eval_model(Xtrain=None,Xtraintilde=Xtrain[modeltype],Xtest=Xtrain[modeltype])
        np.savetxt("data/DAA_results/train_loss_"+modeltype+"_K="+str(K)+'_SSE.txt',daa_train_loss,delimiter=',')
        np.savetxt("data/DAA_results/test_loss_"+modeltype+"_K="+str(K)+'_SEE.txt',daa_test_loss,delimiter=',')