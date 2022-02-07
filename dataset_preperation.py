# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:12:22 2021

@author: Noam
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import sklearn
import sklearn.decomposition
import itertools
from find_temp_script import find_temp
class data_read():
    def __init__(self,label_name,short=False,restrict_samples_n=False,data_already_exist=False,do_over_under_sample=True,filter_conds=False,discard_labels=[],allow_nan_for_feat_percent=1,name_spec=''):
        self.label_name = label_name
        self.discard_labels = discard_labels
        self.do_over_under_sample = False #do_over_under_sample
        self.do_augment = False
        self.filter_conds = filter_conds
        self.allow_nan_for_feat_percent = allow_nan_for_feat_percent 
        self.name_spec = name_spec # postfix name for the saved dataset
        if not data_already_exist:
            #excel_path = 'C:\Users\Noam\Desktop\READMISSION\NOAM_wo_ID_short.xlsx'
            if restrict_samples_n and restrict_samples_n<10000:
                excel_path = 'NOAM_wo_ID_shrt.xlsx'
            else:
                excel_path = 'NOAM_wo_ID.xlsx'    
            data = pd.read_excel(io=excel_path, header=0, index_col=0).replace('.',np.nan).replace(' ',np.nan)
            with open('full_dataset.pkl', 'wb') as handle:
                data['temper'] = find_temp(data['Date of delivery2'])
                data['year'] = data.apply(lambda x: x['Date of delivery2'].year,axis=1)
                data['month'] = data.apply(lambda x: x['Date of delivery2'].month,axis=1)
                data['dayofweek'] = data.apply(lambda x: x['Date of delivery2'].dayofweek-1,axis=1)
                pickle.dump(data,handle)
        else:
            print('Note: load data from pkl file and not from excel file!')
        with open('full_dataset.pkl', 'rb') as handle:           
            data = pickle.load(handle)
        if restrict_samples_n and len(data)>restrict_samples_n:
            data = data.iloc[:restrict_samples_n,:]
        print(f'Data reloaded sucssesfuly (N:{len(data)}). took {round(time.time()-start_time)} sec.')
        self.data=data
        
        

    def preprocess(self,data):
        assert len(data)>0,'empty data'
        orig_data = data.copy()
        name_spec = self.name_spec
        #numerical feat for augmentation
        #features to change from str to int
        '''
        plt.figure()
        data['year'] = data.apply(lambda x: x['Date of delivery2'].year,axis=1)
        data['year_month'] = data.apply(lambda x: '{:%Y-%m}'.format(datetime.datetime.strftime(x['Date of delivery2'],'yy-MM')),axis=1)
        plt.plot(data.iloc[-35000:-30000].groupby('year_month')[self.label_name].mean())
        data['moving_mean100'] = data.rolling(100,on='Date of delivery2')[self.label_name].mean()
        data['moving_mean500'] = data.rolling(500,on='Date of delivery2')[self.label_name].mean()
        data['moving_mean1500'] = data.rolling(1500,on='Date of delivery2')[self.label_name].mean()
        data = data.sort_values('Date of delivery2').set_index('Date of delivery2')
        plt.show()
        data[['moving_mean100','moving_mean500','moving_mean1500']].loc[data.year.isin(range(2015,2022)),:].plot()
        plt.plot(data.groupby('Date of delivery2')[self.label_name].mean())
        '''
        age = ['Maternal age, years']
        feat_in_str = [age,'First_vaginaltest']
        for feat in feat_in_str:
            data[feat] = data[feat].astype(float)
        self.numerical_feat = ['Maternal age, years',
                                            'אורך הלידה כולל שלב לטנטי_דקות'	 ,
                         'אורך הלידה ללא שלב לטנטי_דקות'	,
                            'אורך שלב ראשון_דקות'	,
                            'זמן ירידת מים_דקות'	,
                            'אורך שלב שני_דקות'	,
                            'אורך שלב שלישי_דקות',
]
        self.numerical_feat_limits = np.array([[12,60],
                                               [0,7500],[0,7500],[0,7500],
                                               [0,6*7*24*60],[0,7500],[0,7500]])
        del_samples = []
        for i,feat in enumerate(self.numerical_feat):

            below_min =data.loc[data[feat]<self.numerical_feat_limits[i,0],feat].copy()
            above_max =data.loc[data[feat]>self.numerical_feat_limits[i,1],feat].copy()

            outliers_low_values = below_min.values
            outliers_high_values = above_max.values
            del_samples.extend([*below_min.index, *above_max.index])
            #data.loc[[*below_min.index, *above_max.index],feat]=np.nan
            minval = data[feat].min()
            maxval = data[feat].max()
            '''
            data.loc[below_min.index,feat]=minval*1e11
            data.loc[above_max.index,feat]=maxval
            '''
            print(f'{feat}:found the following values\n')
            if len(above_max)>0:
                print(f' HIGH:{outliers_high_values}')
            else:
                    print('no High values was found\n')
            if len(below_min)>0:
                print(f'LOW {outliers_low_values}')
            else:
                    print('no Low values was found\n')
        print(data.loc[pd.Series(del_samples),self.numerical_feat])
        #data[self.numerical_feat[0]] = data[self.numerical_feat[0]].apply(lambda x: float(x)-100 if float(x)>100 else float(x))
        find_std = lambda x:pd.to_numeric(data.loc[pd.to_numeric(data[x], errors='coerce').notnull(),x]).std()
        self.numerical_feat_std = [find_std(x) for x in self.numerical_feat]
        [print(self.numerical_feat[i]+' SD:'+str(round(self.numerical_feat_std[i],2))) for i in range(len(self.numerical_feat_std))]
        assert any(self.numerical_feat_std)<600,'problem with std'
        #features engeneering 
        eng_feats=['labor_complxity_axis1','labor_complxity_axis2']     
        labor_complxity_feats = [
         'אורך הלידה כולל שלב לטנטי_דקות',
          'אורך הלידה ללא שלב לטנטי_דקות',
'אורך שלב ראשון_דקות',
             'זמן ירידת מים_דקות',
           'אורך שלב שני_דקות'	,
            'אורך שלב שלישי_דקות'
            ]
        pca = sklearn.decomposition.PCA(n_components=2)
        data['labor_complxity_axis1']=np.nan
        data['labor_complxity_axis2']=np.nan
        new_labor_axis_for_knowh_times = pca.fit_transform(data[labor_complxity_feats].dropna().copy())
        data.loc[data[labor_complxity_feats].dropna().index,'labor_complxity_axis1'] = new_labor_axis_for_knowh_times[:,0]
        data.loc[data[labor_complxity_feats].dropna().index,'labor_complxity_axis2'] = new_labor_axis_for_knowh_times[:,1]
        print(f'labor pca explin:{pca.explained_variance_ratio_} of the variance of {labor_complxity_feats}')
        
        #features to replace nan in 0
        feat_to_nan_zero = ......
        data[feat_to_nan_zero]  = data[feat_to_nan_zero].replace(np.nan,int(0))


        #features to replace nan in -1e7
        feat_to_0 = [.......]
        #data[feat_to_0]  = data[feat_to_realmean].replace(np.nan,-1e7)
        data[feat_to_0]  = data[feat_to_0].replace(np.nan,0)
        #features to replace nan in median
        feat_to_med = ['temper','Hospitalization length, days ','Hemoglobin drop, gram/dl','Hemoglobin Ba Kabala','First_vaginaltest']
        for feat in feat_to_med:
            data[feat]  = data[feat].replace(np.nan,data[feat].median())
        #weight will be replaced nan or 0 in week- specific median
        feat_to_med = [...]
        for gender,twins,week in itertools.product(['0','1'],['0','1'],data['Gestational age at delivery'].unique()):
            if not np.isnan(week):
                specific_class_cond = '`Male gender`=='+gender+' and `Multifetal gestation`=='+twins+' and `Gestational age at delivery`=='+str(week)
                specific_class_cond_idx = data.query(specific_class_cond).index
                class_infant_weight_median = data.loc[specific_class_cond_idx,'משקל'].median()
                data.loc[specific_class_cond_idx,'weight']  = data.loc[specific_class_cond_idx,'weight'].replace(np.nan,class_infant_weight_median)
                data.loc[specific_class_cond_idx,'weight']  = data.loc[specific_class_cond_idx,'weight'].replace(0,class_infant_weight_median)
        #catgorical values, replace NaN with mode
        data['Blood Type'] = data['Blood Type'].replace(np.nan,data['Blood Type'].mode()[0])
        '''
        plot_age_dif_hist = False
        if plot_age_dif_hist:
            age_diff_np = np.array([a3-a4 for a3,a4 in zip(data.loc[1:,spouse_age].to_numpy(),data.loc[1:,age].to_numpy())])
            plt.hist(age_diff_np,range=(-6,10),bins=20);plt.show()
            mean_age_diff =  np.nanmean(age_diff_np)
            print('mean diffrence between age of husbend and wife {:.2f} SD:{:.2f}'.format(mean_age_diff,np.nanstd(age_diff_np)))
        else:
            mean_age_diff = 1.94 
        
        data[spouse_age] = data.apply(lambda x:float(x[age] + mean_age_diff) if pd.isna(x[spouse_age]) else x[spouse_age],axis='columns')
        '''
        feat_for_learning = pd.read_excel(io='cols_for_learning.xlsx')

        feat_for_learning =  pd.DataFrame([],columns=[feat for feat in feat_for_learning.columns if 'Unnamed' not in feat])
        #assert not np.any(['Unnamed' in a for a in  feat_for_learning.columns.tolist()]),'Excel File with empty column'
        feat_for_learning.insert(loc=15,column=eng_feats[0],value=np.nan)
        feat_for_learning.insert(loc=16,column=eng_feats[1],value=np.nan)

        feat_table = pd.DataFrame((),columns=data.columns)
        orig_data_with_impute=data.copy()
        data = self.filter_data(data)
        if self.do_over_under_sample:
            #data['augment_grp']=0
            feat2take = [*feat_for_learning,self.label_name,'augment_grp']
            data = self.over_under_sample(data)
        else:
            feat2take = [*feat_for_learning,self.label_name]
        data = data.loc[:,~ data.columns.isin(self.discard_labels)]
        feat_bin = [(i in feat2take) for i in data.columns]  
        nan_in_col = data.isna().sum().to_numpy()
        assert len(data)>0,'empty data'
        samples_n = len(data)
        allow_nan_for_feat_percent = self.allow_nan_for_feat_percent
        allow_nan_for_feat_thrs = allow_nan_for_feat_percent*samples_n/100
        assert len(feat_bin)==len(nan_in_col), f'PROBLEM found: feat:{[a for a in data.columns if a not in feat2take[feat_bin]]} not in feat_bin:'
        pass_criterion = np.logical_and(feat_bin, nan_in_col<allow_nan_for_feat_thrs)
        assert len(feat_bin)==len(data.columns)
        assert len(pass_criterion)==len(data.columns)
        for i in range(len(data.columns)):
            print(f'{data.columns[i]}: taken? {pass_criterion[i]}. Empty N:{nan_in_col[i]} ({round(nan_in_col[i]*100/samples_n,2)}%)')
        label = data[self.label_name]
        data_wo_drop = data.copy() #coping data and labels for aditional pickle saving without nan dropping for catBoost classifier
        label_wo_drop = label.copy()
        data = data.iloc[:,pass_criterion]
        droped_samples_index = data.isna()
        droped_samples_index = droped_samples_index.any(axis=1)
        droped_samples_index = data[droped_samples_index].index
        del_samples.extend(droped_samples_index)
        print(data.loc[data.query('hospital_is_SZ==0').index,self.label_name].value_counts())
        print(f'Overall {len(np.where(np.array(del_samples)==False)[0])} were droped from dataset \n{np.where(np.array(del_samples)==False)}')
        #data = data.dropna()
        del_samples = pd.Series(del_samples)
        droped_data = orig_data_with_impute.loc[del_samples[del_samples.isin(data.index)],:]
        excel_filename='droped_records_'+name_spec
        with open(excel_filename+'.pkl', 'wb') as handle:
            pickle.dump(droped_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        droped_data.to_excel(excel_filename+'.xlsx')
        data = data.drop(del_samples[del_samples.isin(data.index)],axis=0)
        print(f'\n\nDiscard samples with nan values for features with less than {allow_nan_for_feat_percent}%')
        print(f'Samples reduces from {samples_n} to {len(data)}')
        # new features - 
        weight_frac = np.argsort(data['weight']) /len(data)
        data['weight_percent_abs'] = np.abs(weight_frac-0.5)*100
        gest_frac = np.argsort(data['Gestational age at delivery']) /len(data)
        data['Gestational_age_at_delivery_percent_abs'] = np.abs(gest_frac-0.5)*100
        data['year'] = data.apply(lambda x: x['Date of delivery2'].year,axis=1)
        
        #data.pop('Date of delivery2')
        
        label = data[self.label_name]
        data['label'] = label
        if self.label_name in data.columns:
            data.pop(self.label_name)
        if self.do_over_under_sample and self.do_augment:
                filename = 'over_under_aug_dataset'+name_spec+'.pkl'
        elif self.do_over_under_sample:
            filename = 'over_under_sampled_preproccesed_dataset'+name_spec+'.pkl'
        else:
            filename = 'preproccesed_dataset'+name_spec+'.pkl'
        self.save(filename,label,data)
        filename1 = filename
        
        #### dataset with categorical features
        cat_feats = pd.read_excel(io='categorical features.xlsx')
        cat_feats =  pd.DataFrame([],columns=[feat for feat in cat_feats.columns if 'Unnamed' not in feat])
      
        assert not np.any(['Unnamed' in a for a in  cat_feats.columns.tolist()]),'Excel File with empty column'
        discard_cat_feats = cat_feats.columns.isin([self.label_name,*self.discard_labels])
        cat_feats = cat_feats.columns[~discard_cat_feats]
        feat_bin = [(i in [*feat_for_learning,*cat_feats,self.label_name,'augment_grp']) for i in data_wo_drop.columns]
        allow_nan_for_cat_feats = [i in cat_feats for i in data_wo_drop.columns]
        nan_in_col = data_wo_drop.isna().sum().to_numpy()
        pass_criterion = np.logical_or(allow_nan_for_cat_feats,np.logical_and(feat_bin, nan_in_col<allow_nan_for_feat_thrs))
        assert len(feat_bin)==len(data_wo_drop.columns)
        for i in range(len(data.columns)):
            print(f'{data_wo_drop.columns[i]}: taken? {pass_criterion[i]}. Empty N:{nan_in_col[i]} ({round(nan_in_col[i]*100/samples_n,2)}%)')
        for feat in cat_feats:
            data_wo_drop[feat] = data_wo_drop[feat].replace(np.nan,-1)
        data_wo_drop = data_wo_drop.iloc[:,pass_criterion]
        droped_samples_index = data_wo_drop.isna().any(axis='columns')
        droped_samples_index = droped_samples_index==False
        data_wo_drop = data_wo_drop.dropna()
        data_wo_drop = data_wo_drop.drop(del_samples[del_samples.isin(data_wo_drop.index)],axis=0)
        
        weight_frac = np.argsort(data_wo_drop['weight']) /len(data)
        data_wo_drop['weight_percent_abs'] = np.abs(weight_frac-0.5)*100
        gest_frac = np.argsort(data_wo_drop['Gestational age at delivery']) /len(data)
        data_wo_drop['Gestational_age_at_delivery_percent_abs'] = np.abs(gest_frac-0.5)*100
        data_wo_drop['year'] = data_wo_drop.apply(lambda x: x['Date of delivery2'].year,axis=1)
        #data_wo_drop.pop('Date of delivery2')
        print(f'\n\ncat Discard samples with nan values for features with less than {allow_nan_for_feat_percent}%')
        print(f'cat Samples reduces from {samples_n} to {len(data)}')
        label_wo_drop = label_wo_drop.loc[droped_samples_index]
        data_wo_drop['label'] = label_wo_drop
        if self.label_name in data_wo_drop.columns:
            data_wo_drop.pop(self.label_name)
       
        filename = filename[:-4]+'_cat.pkl'

        assert len(data_wo_drop)>0
        self.save(filename,label_wo_drop,data_wo_drop)
        for file in [filename1,filename]:
            print('='*30)
            print(f'dataset {file} has been created!')
            print('='*30)
        print(f'feats:{data_wo_drop.columns}')
        return data,label
    def over_under_sample(self,data):
        over_sample_rate = 2
        if over_sample_rate>1:
            data['augment_grp'] = np.arange(len(data))
        pos_samples_wo_over_sample = data.loc[data[self.label_name]==1,:]
        pos_samples = pos_samples_wo_over_sample.copy()
        if over_sample_rate==0:
            print('Just *under* samples without oversample!') 
        else:
            for i in np.arange(np.ceil(over_sample_rate)):
                pos_samples_oversample_wo_aug = pos_samples_wo_over_sample.copy().sample(frac=min(1,np.abs(over_sample_rate-i)),replace=False)
                if self.do_augment:
                    temp_pos_samples = pos_samples_oversample_wo_aug.apply(lambda sample:self.augment(sample),axis=1)
                else:
                    temp_pos_samples = pos_samples_oversample_wo_aug
                pos_samples = pd.concat((temp_pos_samples,pos_samples))
            for feat in self.numerical_feat:
                
                pd.DataFrame(pos_samples_wo_over_sample[feat]).plot(kind='density')
                plt.show()
                plt.xlabel(feat+' before aug')
                pd.DataFrame(pos_samples[feat]).plot(kind='density')
                plt.xlabel(feat+' after aug')
                plt.show()
            pos_samples  = pos_samples.sample(frac=1)
        neg_samples = data.loc[data[self.label_name]==0,:]
        neg_samples = neg_samples.sample(frac=len(pos_samples)/len(neg_samples))
        if self.do_augment:
            neg_samples = neg_samples.apply(lambda sample:self.augment(sample),axis=1)
        return_data = pd.concat((pos_samples,neg_samples)).sample(frac=1)
        return return_data
    def augment(self,sample):
        for i,feat in enumerate(self.numerical_feat):
            rand_std = self.numerical_feat_std[i]/15
            if sample[feat]>0:
                sample[feat] += np.random.normal(scale=rand_std)
                if sample[feat]<0:
                    sample[feat]=0
        return sample
    def save(self,filename,label,data):
       print('Imbalance {:.2f}% ({:f}) / {:.2f}% ({:f}) - Label:{} / NO'.format(
           label.mean()*100,
           label.sum(),
           ((1-label.mean())*100),
           len(label)-label.sum(),
           self.label_name))
       data_SZ = self.filter_data(data.copy(),'hospital_is_SZ==1')
       data_BH = self.filter_data(data.copy(),'hospital_is_SZ==0')
       print(f'features:{data.columns}')
       with open('SZ_'+filename, 'wb') as handle:
           pickle.dump(data_SZ, handle, protocol=pickle.HIGHEST_PROTOCOL)
       print(f'Dataset {"SZ_"+filename} has been saved')
       with open('BH_'+filename, 'wb') as handle:
           pickle.dump(data_BH, handle, protocol=pickle.HIGHEST_PROTOCOL)
       print(f'Dataset {"BH_"+filename} has been saved')
       
       with open(filename, 'wb') as handle:
           pickle.dump(data.drop('hospital_is_SZ',axis=1), handle, protocol=pickle.HIGHEST_PROTOCOL)
       print(f'Dataset {filename} has been saved')
    def filter_data(self,data,filter_conds=False):
       if filter_conds:
           filter_conds =[filter_conds] if isinstance(filter_conds,str) else filter_conds
           for cur_filter in filter_conds:
               print(f'Filter by {cur_filter} N before:{len(data)}',end='')
               data = data.query(cur_filter)
               print(f', N after:{len(data)}.')

       elif self.filter_conds:
           return self.filter_data(data,self.filter_conds)
       return data
start_time = time.time()
if __name__ == "__main__":  
    #tree = data_read(label_name=90)
    discard_labels=[...]
    tree = data_read(data_already_exist=True,label_name='In labor cesarean',
                     filter_conds=['...==1', '...==0','...==0','...==0',' ...==0'],
                     discard_labels=discard_labels,name_spec='+temp')
    data,label =  tree.preprocess(tree.data)
    
    tree = data_read(data_already_exist=True,label_name='In labor cesarean',
                     filter_conds=['...==1', '...==0','...==0','...==0',' ...==0'],
                     discard_labels=[*discard_labels,'temper','month','dayofweek'])
    data,label =  tree.preprocess(tree.data)


    data_load_time=time.time()
    end = time.time() 
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Overall time:',start_time,end)
    print(data.columns)
