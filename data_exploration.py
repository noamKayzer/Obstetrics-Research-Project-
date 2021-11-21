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
%matplotlib qt
tree = clf(label_name=90,model_type='cat',filename='preproccesed_dataset_cat.pkl')

x,y = tree.data_read(split_train_test=False)
print(x.describe())
x['label']=y
#for i in [42,30,60]:
#    x[str(i)]=x[i]
non_object_feats = [str(feat) for feat,a in zip(x.columns[6:],x.dtypes[6:].tolist()) if a != 'object']
non_object_feats.append('label')
shorten = lambda feat : feat if len(feat)<10 else feat[:11]
short_non_object_feats = [shorten(feat) for feat in non_object_feats]
#corrcoef = np.corrcoef(x[non_object_feats].T)
corrcoef = x[non_object_feats].corr()
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(np.corrcoef(corrcoef))

ax.set_xticks(np.arange(len(non_object_feats)))
ax.set_yticks(np.arange(len(non_object_feats)))

ax.set_xticklabels(short_non_object_feats)
ax.set_yticklabels(short_non_object_feats)
ax.set_title("diagonisis covariance matrix")
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
 rotation_mode="anchor")
fig.tight_layout()
plt.show()
plt.figure(figsize=(30,30))
sns.heatmap(x[non_object_feats].corr(),
    #annot=True,
    linewidths=.5,
    center=0,
    cbar=False,
    cmap="YlOrBr",
    xticklabels = short_non_object_feats,
    yticklabels = short_non_object_feats
    )
plt.show()