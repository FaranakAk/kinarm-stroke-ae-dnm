# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:45:44 2020

@author: fakbarifar


Builds paired (R/L) Kinarm feature rows, keeps first reads, aligns to CMSA, and saves .npz files.




"""
#%%
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from datetime import datetime

from Plots import data_time_dist


#%%
def keep_1st_read(xData0, xDate0, xTime0):
    # Created to remove later exams for subjects with multiple data recordings
    # first reading in 90 days since stroke
    # In data_time_dist, subjects that don't have a stroke date in lesion_locations.csv are omitted too.
    
    x, dates, x_sb, nodate_subs, sub_for_1stRead = data_time_dist(xData0, xDate0)
    
    xData = []
    xDate = []
    xTime = []
    repINDs = []
    for i in range(int(len(xData0))):
        if ((xData0[i, 0] not in repINDs) & (xData0[i, 0] in sub_for_1stRead)):
            inds_each_sub = np.where(xData0[:, 0]==xData0[i, 0])[0]
            dates_each_sub = []
            for j in range(int(len(inds_each_sub))):
                dates_each_sub.append(datetime.strptime(xDate0[inds_each_sub[j]], "%d/%m/%Y"))
            
            DT_xDate0 = []
            for k in range(int(len(xDate0))):
                DT_xDate0.append(datetime.strptime(xDate0[k], "%d/%m/%Y"))
                
            minDate_each_sub = min(dates_each_sub)
            inds_save = np.where((xData0[:, 0]==xData0[i, 0]) & (np.asarray(DT_xDate0)==minDate_each_sub))[0]
            xData.append(xData0[inds_save[0], :])
            xData.append(xData0[inds_save[1], :])
            
            xDate.append(xDate0[inds_save[0]])
            xDate.append(xDate0[inds_save[1]])
            xTime.append(xTime0[inds_save[0]])
            xTime.append(xTime0[inds_save[1]])
            
            repINDs.append(xData0[inds_save[0], 0])
            repINDs.append(xData0[inds_save[1], 0])
 
            
    xData = np.asarray(xData)
    xDate = np.asarray(xDate)
    xTime = np.asarray(xTime)
        
    return xData, xDate, xTime



    
    

#%% Reading the CMSA score from the csv files based on the closest date to KINARM data (before train, val, test split)

def PrepData_CMSA_1(X_all, X_date, X_time, X_labels, method='arm'):       
    clncPATH = "C:/Users/fakbarifar/OneDrive - Queen's University/Faranak/stroke data/clinical"
    clncFile = os.path.join(clncPATH, "clinical_scores.csv")
    df = pd.read_csv(clncFile)    
    
    matlab_data = sio.loadmat(os.path.join(clncPATH, 'subjects_missing_clinical_ax.mat') )
    
    tTS = X_all[:,3]
    
    CMSA = []
    
    TPA = []
    HEMINEGLECT = []
    AFFECTEDARM = []
    WEARSGLASSES = []
    PTONGOING = []
    APHASIA = []
    HEIGHT = []
    WEIGHT = []
    FOLSTEIN = []
    BITSCORE = []
    VISUALFIELDDEFICIT = []
    PURDUE = []
    DYNAPINCH = []
    THUMB = []
            
    noIND = []
    for cc in range(len(X_all)):
            
        date_kinarm = X_date[cc]
        subID = X_all[cc,0]
        RorL = X_all[cc,1]
        
        if RorL==1:
            RorL = 'RIGHT'
        elif RorL==2:
            RorL = 'LEFT'
            
                    
        date_kinarm = datetime.strptime(date_kinarm, "%d/%m/%Y")
        
        noCMSA = matlab_data['sk_missing_clin']
        if np.any(noCMSA==subID):
            noIND.append(cc)
            continue
            
            
        if X_labels[cc]==1:
            ind_clnc = np.where(df['SUBJECTKEY']==subID)
            minDays = 10000
            for aa in ind_clnc[0]:
                date_clnc = df['CLINSTARTDATE'][aa]
                date_clnc = datetime.strptime(date_clnc, "%d/%m/%Y")
                if abs((date_kinarm-date_clnc).days)<minDays:
                    date_clnc_mindist = date_clnc
                    minDays = abs((date_kinarm-date_clnc).days)
                    
            date_clnc_mindist = datetime.strftime(date_clnc_mindist, "%d/%m/%Y")
            ind_clnc_mindist = np.where((df['SUBJECTKEY']==subID) & (df['CLINSTARTDATE']==date_clnc_mindist))
                  
              
            if RorL=='RIGHT':
                if (np.mean(df['CHEDOKERIGHTHAND'][ind_clnc_mindist[0]])>0) & (np.mean(df['CHEDOKERIGHTARM'][ind_clnc_mindist[0]])>0): # to remove (-2)/(-3) values in clinical scores
                    if method=='average':
                        CMSA.append(np.round(np.mean(df['CHEDOKERIGHTHAND'][ind_clnc_mindist[0]] + df['CHEDOKERIGHTARM'][ind_clnc_mindist[0]])/2))
                    elif method=='sum':
                        CMSA.append(np.mean(df['CHEDOKERIGHTHAND'][ind_clnc_mindist[0]] + df['CHEDOKERIGHTARM'][ind_clnc_mindist[0]]))
                    elif method=='min':
                        CMSA.append(np.min([np.mean(df['CHEDOKERIGHTHAND'][ind_clnc_mindist[0]]), np.mean(df['CHEDOKERIGHTARM'][ind_clnc_mindist[0]])]))
                    elif method=='hand':
                        CMSA.append(np.mean(df['CHEDOKERIGHTHAND'][ind_clnc_mindist[0]]))
                    elif method=='arm':
                        CMSA.append(np.mean(df['CHEDOKERIGHTARM'][ind_clnc_mindist[0]]))
                        
                        PURDUE.append(df['PURDUERIGHT'][ind_clnc_mindist[0]])
                        DYNAPINCH.append(df['DYNAPINCHRIGHT'][ind_clnc_mindist[0]])
                        THUMB.append(df['THUMBRIGHT'][ind_clnc_mindist[0]])
                        
                        TPA.append(df['TPA'][ind_clnc_mindist[0]])
                        HEMINEGLECT.append(df['HEMINEGLECT'][ind_clnc_mindist[0]])
                        AFFECTEDARM.append(df['AFFECTEDARM'][ind_clnc_mindist[0]])
                        WEARSGLASSES.append(df['WEARSGLASSES'][ind_clnc_mindist[0]])
                        PTONGOING.append(df['PTONGOING'][ind_clnc_mindist[0]])
                        APHASIA.append(df['APHASIA'][ind_clnc_mindist[0]])
                        HEIGHT.append(df['HEIGHT'][ind_clnc_mindist[0]])
                        WEIGHT.append(df['WEIGHT'][ind_clnc_mindist[0]])
                        FOLSTEIN.append(df['FOLSTEIN'][ind_clnc_mindist[0]])
                        BITSCORE.append(df['BITSCORE'][ind_clnc_mindist[0]])
                        VISUALFIELDDEFICIT.append(df['VISUALFIELDDEFICIT'][ind_clnc_mindist[0]])
                        
                else:
                    noIND.append(cc)
            elif RorL=='LEFT':
                if (np.mean(df['CHEDOKELEFTHAND'][ind_clnc_mindist[0]])>0) & (np.mean(df['CHEDOKELEFTARM'][ind_clnc_mindist[0]])>0):
                    if method=='average':
                        CMSA.append(np.round(np.mean(df['CHEDOKELEFTHAND'][ind_clnc_mindist[0]] + df['CHEDOKELEFTARM'][ind_clnc_mindist[0]])/2))
                    elif method=='sum':
                        CMSA.append(np.mean(df['CHEDOKELEFTHAND'][ind_clnc_mindist[0]] + df['CHEDOKELEFTARM'][ind_clnc_mindist[0]]))
                    elif method=='min':
                        CMSA.append(np.min([np.mean(df['CHEDOKELEFTHAND'][ind_clnc_mindist[0]]), np.mean(df['CHEDOKELEFTARM'][ind_clnc_mindist[0]])]))
                    elif method=='hand':
                        CMSA.append(np.mean(df['CHEDOKELEFTHAND'][ind_clnc_mindist[0]]))
                    elif method=='arm':
                        CMSA.append(np.mean(df['CHEDOKELEFTARM'][ind_clnc_mindist[0]]))
                        
                        PURDUE.append(df['PURDUELEFT'][ind_clnc_mindist[0]])
                        DYNAPINCH.append(df['DYNAPINCHLEFT'][ind_clnc_mindist[0]])
                        THUMB.append(df['THUMBLEFT'][ind_clnc_mindist[0]])
                        
                        TPA.append(df['TPA'][ind_clnc_mindist[0]])
                        HEMINEGLECT.append(df['HEMINEGLECT'][ind_clnc_mindist[0]])
                        AFFECTEDARM.append(df['AFFECTEDARM'][ind_clnc_mindist[0]])
                        WEARSGLASSES.append(df['WEARSGLASSES'][ind_clnc_mindist[0]])
                        PTONGOING.append(df['PTONGOING'][ind_clnc_mindist[0]])
                        APHASIA.append(df['APHASIA'][ind_clnc_mindist[0]])
                        HEIGHT.append(df['HEIGHT'][ind_clnc_mindist[0]])
                        WEIGHT.append(df['WEIGHT'][ind_clnc_mindist[0]])
                        FOLSTEIN.append(df['FOLSTEIN'][ind_clnc_mindist[0]])
                        BITSCORE.append(df['BITSCORE'][ind_clnc_mindist[0]])
                        VISUALFIELDDEFICIT.append(df['VISUALFIELDDEFICIT'][ind_clnc_mindist[0]])

            
            
                else:
                    noIND.append(cc)
                    
                    
            
            
        else:
            if method=='average':
                CMSA.append(np.round(np.mean(float(7) + float(7))/2))
            elif method=='sum':
                CMSA.append(np.mean(float(7) + float(7)))
            elif method=='min':
                CMSA.append(np.mean(float(7)))
            elif method=='hand':
                CMSA.append(np.mean(float(7)))
            elif method=='arm':
                CMSA.append(np.mean(float(8)))
            
    CMSA = np.array(CMSA)
    X_all = np.delete(X_all, noIND, 0)
    X_date = np.delete(X_date, noIND, 0)
    X_time = np.delete(X_time, noIND, 0)
    X_labels = np.delete(X_labels, noIND, 0)
    
    clnc_hdr = np.concatenate([np.expand_dims(TPA,axis=1), np.expand_dims(HEMINEGLECT, axis=1), np.expand_dims(AFFECTEDARM, axis=1), np.expand_dims(WEARSGLASSES, axis=1), np.expand_dims(PTONGOING, axis=1), np.expand_dims(APHASIA, axis=1), np.expand_dims(HEIGHT, axis=1), np.expand_dims(WEIGHT, axis=1), np.expand_dims(FOLSTEIN, axis=1), np.expand_dims(BITSCORE, axis=1), np.expand_dims(VISUALFIELDDEFICIT, axis=1), np.expand_dims(PURDUE, axis=1), np.expand_dims(DYNAPINCH, axis=1), np.expand_dims(THUMB,axis=1)], axis=1)
    
    
    return X_all, X_date, X_time, X_labels, CMSA, clnc_hdr, noIND 


