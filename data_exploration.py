# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:21:49 2021

@author: Noam
Initiall data visualization: Descriptive statistics and covariance matrix 
"""
# Exploratory Data Analysis-EDA
from main import clf
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from io import StringIO
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import cohen_kappa_score,balanced_accuracy_score,  plot_confusion_matrix,log_loss,roc_auc_score,f1_score,make_scorer,recall_score
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.image as pltimg
from io import StringIO
from sklearn.inspection import permutation_importance
from functools import partial
import time
import pickle
import optuna
import plotly
import plotly.io as pio
pio.renderers.default='svg'
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from io import StringIO
import seaborn as sns
import xlsxwriter
from pdpbox import pdp, get_dataset, info_plots
# %matplotlib qt
from dataset_preperation import data_read
from dateutil.relativedelta import relativedelta
import datetime
discard_labels=['Ceasarean',
                                     'Hospitalization length, days ' ,
                                     'prolonged hospital stay','Vaginal tear',
                                     'labor_complxity_axis1',
                                     'labor_complxity_axis2','POP',
                                                                 'דרגה 1',
                                                'דרגה 2',                               
                                                'דרגה 3',                              
                                                'דרגה 4 ',                              
                                                'Retained placenta/placental fragments  ' ,
                                               	'Perineal tear grade 3/4',
                                                   'Laceration'	,
                                                   'Episiotomy',
                                                   'Hemoglobin drop, gram/dl',
                                                   'DELTA_ֹHB>3',
                                                   'DELTA_ֹHB>4'	,
                                                   'DELTA_ֹHB>5'	,
                                                   'Chorioamionitis'	,
                                                   'Puerperal fever'	,
                                                   'Hexakapron'	,
                                                   'Parenteral Iron administration',	
                                                   'Blood products transfusion',
                                                   'Hysterectomy	laparotomy',
                                                   'Failed Vacuum',	'Vaccuum',
                                                   '1-Minute Apgar score < 7',
                                                   '5-Minute Apgar score < 7'	,
                                                   'NICU admission'	,
                                                   'Meconium aspiration syndrome'	,
                                                   'Jaundice'	,
                                                   'TTN'	,
                                                   'Mechanical ventilation'	,
                                                   'Seizures'	,
                                                   'Erb’s palsy/fracture of clavicle'	,
                                                   'Hypoglycemia'	,
                                                   'Sepsis'	,
                                                   'Encephalopathy'	,
                                                   'Intracranial hemorrhage'	,
                                                   'Birth asphyxia',
                                                   'Prolonged hospital stay',
                                                   'Hysterectomy', 'laparotomy',
           'shoulder dystocia', 'Past Shoulder Dystocia',
           'IUFD@Intrauterine Fetal Death@Antepartum Fetal Death@INTRAUTERINE DEATH@Cause of Death Unknown@Early Neonatal Death',
           'Maternal ICU admissions ', 'Postpartum hemorrhage','GBS',
    'Birthweight >4000 grams ',
    'ANEMIA_11',
    'ANEMIA_10',
    'ANEMIA_9',
    'ANEMIA_8',
    'Gestational age at delivery<37 week ',
    
    'LGA',
    'SGA',
                                     'אורך הניתוח הקיסרי'	,
                                     'אורך הניתוח הקיסרי-מתחילהעדלידה',
                                     'אורך הלידה כולל שלב לטנטי_דקות',
                                     'אורך הלידה ללא שלב לטנטי_דקות',
           'אורך שלב ראשון_דקות',
           'זמן ירידת מים_דקות', 
           'אורך שלב שני_דקות',
        'אורך שלב שלישי_דקות',
    ]
label_name = 'In labor cesarean'
tree = data_read(data_already_exist=True,label_name='In labor cesarean',
                 filter_conds=['Primipara==1', 'Cesarean_Section_No_Trial_of_Labor==0','Home_or_car_delivery==0','Elective_CS==0',' `Non-vertex presentation`==0'],
                 discard_labels=discard_labels)
data,label =  tree.preprocess(tree.data)
data['label']=label
def plot_by_time(data,time):
    if time not in data.columns:
        data[time] = data.apply(lambda x: getattr(x['Date of delivery2'],time),axis=1)
    grouped = data.groupby(time).label
    data.groupby(time).label.mean().plot()
    plt.title(time)
    plt.show()
    '''
    plt.plot(data[time].unique(),grouped.mean())
    plt.title(time)
    plt.show()
    plt.errorbar(y=grouped.mean(),yerr=grouped.std(),x=data[time].unique())
    plt.title(time)
    plt.show()
    '''
t0 = data['Date of delivery2'].iloc[0]
time_from_t0 = lambda t:(t.year-t0.year)*12+t.month-t0.month
data['month_from_t0'] = data.apply(lambda x: time_from_t0(x['Date of delivery2']),axis=1)
plot_by_time(data, 'month_from_t0')
plt.figure()
plot_by_time(data, 'dayofweek')
plot_by_time(data, 'year')
plot_by_time(data, 'month')
data.fillna(0).set_index('month_from_t0').sort_values('Date of delivery2').label.rolling(7500).mean().plot()
data['year'] = data.apply(lambda x: x['Date of delivery2'].year,axis=1)

data['moving_mean100'] = data.rolling(100,on='Date of delivery2').label.mean()
data['moving_mean500'] = data.rolling(500,on='Date of delivery2').label.mean()
data['moving_mean1500'] = data.rolling(1500,on='Date of delivery2').label.mean()
data = data.sort_values('Date of delivery2').set_index('Date of delivery2')
plt.show()
data[['moving_mean100','moving_mean500','moving_mean1500']].loc[data.year==2020,:].plot()
plt.show()
plt.plot(data.groupby('Date of delivery2').label.mean())

tree = clf(label_name=label_name,model_type='cat',filename='preproccesed_dataset_cat.pkl',save_prefix='full_dataset')

x,y = tree.data_read(split_train_test=False)
a=x.describe()
a=a.append(pd.DataFrame([x.isin([-1,np.nan,'.']).sum()],index=['NaN']))
workbook = xlsxwriter.Workbook('describe_features.xlsx')
worksheet = workbook.add_worksheet("My sheet")
data=x
data['label']=y
plt.plot(range(2005,2022),data.groupby('year').label.mean())
plt.ylabel('in labor cesearen %')
plt.xlabel('year')
plt.show()
# Iterate over the data and write it out row by row.
est_list = ['name','count','mean','std','min','25%','50%','75%','max','NaN']
for i, cur_estimate in enumerate(est_list):
     worksheet.write(i,0,cur_estimate)
for i,feat in enumerate(a.columns):
    worksheet.write(0, i+1, feat)
    for n in range(len(a[feat])):
        worksheet.write(n+1, i + 1, a[feat][n])
workbook.close()
data_with = x
x['label']=y
data_with['label']=y;data_with = data_with.query('`Induction of labor`>=0')
fig, axes, summary_df = info_plots.target_plot_interact(
    df=data_with, features=['Maternal age, years', 'Induction of labor'], num_grid_points=(18,5),percentile_ranges=[(0.5,99.5),(0,100)],feature_names=['Maternal age, years', 'Induction of labor'],target='label')
fig, axes, summary_df = info_plots.target_plot_interact(
    df=data_with, features=['Maternal age, years', 'Oxytocin augmentation of labor'], num_grid_points=(18,5),percentile_ranges=[(0.5,99.5),(0,100)],feature_names=['Maternal age, years', 'of labor'],target='label')
fig, axes, summary_df = info_plots.target_plot_interact(
    df=data_with, features=['Gestational age at delivery', 'Induction of labor'], num_grid_points=(18,5),percentile_ranges=[(0.5,99.5),(0,100)],feature_names=['Gestational age at delivery', 'Induction of labor'],target='label')
fig, axes, summary_df = info_plots.target_plot_interact(
    df=data_with, features=['Gestational age at delivery', 'Oxytocin augmentation of labor'], num_grid_points=(18,5),percentile_ranges=[(0.5,99.5),(0,100)],feature_names=['Gestational age at delivery', 'Induction of labor'],target='label')
#for i in [42,30,60]:
#    x[str(i)]=x[i]
non_object_feats = [str(feat) for feat,a in zip(x.columns[6:],x.dtypes[6:].tolist()) if a != 'object']
non_object_feats.append('label')
shorten = lambda feat : feat if len(feat)<16 else feat[:15]
#corrcoef = np.corrcoef(x[non_object_feats].T)
x_corrcoef = x[non_object_feats]
x_corrcoef = x_corrcoef.loc[:, x_corrcoef.std() >0]
corrcoef = x_corrcoef.loc[:, x_corrcoef.std() >0].corr()
fig, ax = plt.subplots(figsize=(10,10))

ax.set_xticks(np.arange(len(x_corrcoef.columns)))
ax.set_yticks(np.arange(len(x_corrcoef.columns)))

x_corrcoef_short_labels = [shorten(feat) for feat in x_corrcoef]
ax.set_xticklabels(x_corrcoef_short_labels)
ax.set_yticklabels(x_corrcoef_short_labels)
im = ax.imshow(corrcoef)
ax.set_title("diagonisis covariance matrix")
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
 rotation_mode="anchor")
fig.tight_layout()
plt.show()
plt.figure(figsize=(20,20))
sns.heatmap(corrcoef,
    #annot=True,
    linewidths=.5,
    center=0,
    cbar=False,
    cmap="YlOrBr",
    xticklabels = x_corrcoef_short_labels,
    yticklabels = x_corrcoef_short_labels
    )
plt.show()