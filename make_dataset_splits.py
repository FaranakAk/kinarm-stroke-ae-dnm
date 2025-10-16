# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:14:53 2021

@author: fakbarifar


Creates subject-wise train/val/test splits (60/20/20) and writes .npz packs for the classifier.



"""

#%% import libraries
# import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random 
import tensorflow as tf
import pandas as pd
from build_pairs_and_cmsa import PrepData_CMSA_1, keep_1st_read





#%% initialize random seed
my_seed = 3
# my_seed = 8806
# my_seed = 0
np.random.seed(my_seed)

random.seed(my_seed)

tf.random.set_seed(my_seed)
    
      
#%% I wanted to remove subjects with multiple stroke dates
# Run once

date_file_PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
dateFile = os.path.join(date_file_PATH, "lesion_locations.csv")
df = pd.read_csv(dateFile)

PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
Stroke_All_3 = np.load(os.path.join(PATH, 'Stroke_z_distTest.npz'))
stData0 = Stroke_All_3['RepFeat']
stDate0 = Stroke_All_3['RepDate']
stTime0 = Stroke_All_3['RepTime']
PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
Control_All_3 = np.load(os.path.join(PATH, 'Control_z_distTest.npz'))
ctData0 = Control_All_3['RepFeat']
ctDate0 = Control_All_3['RepDate']
ctTime0 = Control_All_3['RepTime']
    
all_subs = df['SUBJECTKEY']
unique_subs, unique_counts = np.unique(all_subs, return_counts=True)
   


remove_subs = []
for i in range(len(unique_subs)):
    if unique_counts[i]>1:
        all_dates = []
        nodate = 0
        ind_date = np.where(unique_subs[i]==all_subs)
        for j in ind_date[0]:
            # aa = ind_date[0][0]
            date_st = df['DATEOFSTROKE'][j]
            if date_st==date_st:
                if df['DATEOFSTROKE_UNKNOWN'][j]!='Y':
                    all_dates.append(date_st)
            else:
                nodate = 1
                
        if len(set(all_dates)) != 1 or nodate==1: # input_list has all identical elements.
            remove_subs.append(unique_subs[i])
            
            
final_remove = []           
for sb in remove_subs:
    ind_sb = np.where(stData0[:, 0]==sb)
    if ind_sb[0].size>0:
        final_remove.append(sb)
        stData0 = np.delete(stData0, ind_sb[0], 0)
        stDate0 = np.delete(stDate0, ind_sb[0], 0)
        stTime0 = np.delete(stTime0, ind_sb[0], 0)
        
        
        
        
# np.save('stData0_4mat.npy', stData0)
# np.save('stDate0_4mat.npy', stDate0)
# np.save('stTime0_4mat.npy', stTime0)

#%% Read data

PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
# Stroke_All_3 = np.load(os.path.join(PATH, 'Stroke_All_3_zeta.npz'))
# stData0 = Stroke_All_3['RepFeat']
# stDate0 = Stroke_All_3['RepDate']
PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
Control_All_3 = np.load(os.path.join(PATH, 'Control_z_distTest.npz'))
ctData0 = Control_All_3['RepFeat']
ctDate0 = Control_All_3['RepDate']
ctTime0 = Control_All_3['RepTime']
#%% Run this to use multiple recordings for stroke and control
# stData = stData0
# ctData = ctData0

#%% Run this to use multiple recordings for control and first recordings for stroke

stData, stDate, stTime = keep_1st_read(stData0, stDate0, stTime0)
ctData = ctData0
ctDate = ctDate0
ctTime = ctTime0

#%% Selection of affected arm for strokes and dominant arm for controls

PATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
dataFile = os.path.join(PATH, 'clinical_scores.csv')
dfc = pd.read_csv(dataFile)
subjects = dfc['SUBJECTKEY']
unq_subjects = np.unique(subjects)
affected = dfc['AFFECTEDARM']

new_stData = []
new_stDate = []
new_stTime = []
for i_st in range(len(stData)):
    if np.where(subjects==stData[i_st,0])[0].size>0:
        aff = affected[np.where(subjects==stData[i_st,0])[0][0]]
        if aff=='R':
            aff=1
        elif aff=='L':
            aff=2
        elif aff=='B':
            aff=1
        if stData[i_st, 1]==aff:
            new_stData.append(stData[i_st, :])
            new_stDate.append(stDate[i_st])
            new_stTime.append(stTime[i_st])
            
new_stData = np.asarray(new_stData)
new_stDate = np.asarray(new_stDate)
new_stTime = np.asarray(new_stTime)


new_ctData = []
new_ctDate = []
new_ctTime = []
for i_ct in range(len(ctData)):
    if ctData[i_ct, 1]==ctData[i_ct, 2]:
        new_ctData.append(ctData[i_ct, :]) 
        new_ctDate.append(ctDate[i_ct])
        new_ctTime.append(ctTime[i_ct])
    elif ctData[i_ct, 2]==3 and ctData[i_ct, 1]==1:
        new_ctData.append(ctData[i_ct, :])
        new_ctDate.append(ctDate[i_ct])
        new_ctTime.append(ctTime[i_ct])
    
new_ctData = np.asarray(new_ctData)
new_ctDate = np.asarray(new_ctDate)
new_ctTime = np.asarray(new_ctTime)
    
np.savez('match_CT', new_ctData=new_ctData, new_ctDate=new_ctDate, new_ctTime=new_ctTime)
np.savez('match_ST', new_stData=new_stData, new_stDate=new_stDate, new_stTime=new_stTime)

#%% calculate CMSA
new_ctLab = np.zeros(new_ctData.shape[0])
new_stLab = np.ones(new_stData.shape[0])

ctData_clnc, ctDate_clnc, ctTime_clnc, ctLab_clnc, ct_CMSA, ct_hdr, ct_noIND = PrepData_CMSA_1(new_ctData, new_ctDate, new_ctTime, new_ctLab, method='arm')
stData_clnc, stDate_clnc, stTime_clnc, stLab_clnc, st_CMSA, st_hdr, st_noIND = PrepData_CMSA_1(new_stData, new_stDate, new_stTime, new_stLab, method='arm')

ctLabs = np.empty((ctLab_clnc.shape[0], 2))
ctLabs[:, 0] = ctLab_clnc
ctLabs[:, 1] = ct_CMSA

stLabs = np.empty((stLab_clnc.shape[0], 2))
stLabs[:, 0] = stLab_clnc
stLabs[:, 1] = st_CMSA
    
    
#    nPATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data"
#    np.savez(os.path.join(nPATH,'control_b4_stratANDnorm'), ctData_clnc=ctData_clnc, ctDate_clnc=ctDate_clnc, ctTime_clnc=ctTime_clnc, ctLabs=ctLabs, ct_hdr=ct_hdr, ct_noIND=ct_noIND) 
#    np.savez(os.path.join(nPATH,'stroke_b4_stratANDnorm'), stData_clnc=stData_clnc, stDate_clnc=stDate_clnc, stTime_clnc=stTime_clnc, stLabs=stLabs, st_hdr=st_hdr, st_noIND=st_noIND) 
    
    
    
#%% stratification (60 train, 20 val, 20 test)

unique, unique_indices = np.unique(ctData_clnc[:,0], return_index=True)

ctData_clnc = ctData_clnc[unique_indices]
ctDate_clnc = ctDate_clnc[unique_indices]
ctTime_clnc = ctTime_clnc[unique_indices]
ctLabs = ctLabs[unique_indices] 


all_data = np.concatenate([ctData_clnc, stData_clnc], axis=0)
all_labels = np.concatenate([ctLabs[:,1], stLabs[:,1]], axis=0)


train_val_data_all, test_data_all, train_val_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.20, random_state=167, stratify=all_labels)
train_data_all, val_data_all, train_labels, val_labels = train_test_split(train_val_data_all, train_val_labels, test_size=0.25, random_state=167, stratify=train_val_labels)


   

        
    



train_data = train_data_all[:, 6:-1]
val_data = val_data_all[:, 6:-1]
test_data = test_data_all[:, 6:-1]
    
        
    
#%% Normalization, when using z-scores
max_mat = np.max(train_data, axis=0)
max_mat_train = np.tile(max_mat, (train_data.shape[0],1))
min_mat = np.min(train_data, axis=0)
min_mat_train = np.tile(min_mat, (train_data.shape[0],1))

train_data = np.divide((train_data-min_mat_train), (max_mat_train-min_mat_train))

max_mat_val = np.tile(max_mat, (val_data.shape[0],1))
min_mat_val = np.tile(min_mat, (val_data.shape[0],1))
val_data = np.divide((val_data-min_mat_val), (max_mat_val-min_mat_val))

max_mat_test = np.tile(max_mat, (test_data.shape[0],1))
min_mat_test = np.tile(min_mat, (test_data.shape[0],1))
test_data = np.divide((test_data-min_mat_test), (max_mat_test-min_mat_test))
    
#%%
nPATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Current work/data/desom"
np.savez(os.path.join(nPATH, 'train_desom_balanced'), train_data=train_data, train_labels=train_labels)
np.savez(os.path.join(nPATH, 'val_desom_balanced'), val_data=val_data, val_labels=val_labels)
np.savez(os.path.join(nPATH, 'test_desom_balanced'), test_data=test_data, test_labels=test_labels)
   
 
    

