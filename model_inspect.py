# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:54:06 2021

@author: Noam

This function use already trained models and visualize scores-
 ROC, confusion matrices, shap values, tree visualization (for non-ensemble classifier)
"""

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
from  sklearn.utils.class_weight import compute_sample_weight
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


class clf():
    def __init__(self,label_name,model_path, model_type='DT',dataset_filename='preproccesed_dataset.pkl',save_prefix=''):
        self.model_type = model_type
        self.filename = dataset_filename
        self.label_name = label_name
        
        self.save_prefix = save_prefix
        self.multi_obejctive = True
        self.load(model_path)
        
        
    def data_read(self,short=False,restrict_samples_n=False):
        with open(self.filename, 'rb') as handle:
            data = pickle.load(handle)
        if restrict_samples_n:
            data = data.iloc[:restrict_samples_n,:]
        if self.label_name in data.columns and not 'label' in data.columns:
            data['label'] = data[self.label_name]
        if self.label_name in data.columns:
                data.pop(self.label_name)
        if 'cat' in self.filename:
            self.cat_feats = pd.read_excel(io='categorical features.xlsx').columns.tolist()
            for feat in self.cat_feats:
                if data[feat].dtypes == 'float64':
                    data[feat] =data[feat].astype('int64')
        label = data['label']
        data.pop('label')
        print('Imbalance {:.2f}% ({:d}) / {:.2f}% ({:d}) - Yes / NO'.format(
            label.mean()*100,
            label.sum(),
            ((1-label.mean())*100),
            len(label)-label.sum()))
        self.imbalance_rate = label.sum()/len(label)
        if np.abs(self.imbalance_rate-0.5)<0.1:
            #self.scorer = make_scorer(balanced_accuracy_score,greater_is_better=True)#recall_score,average='macro')
            #self.scorer = make_scorer(recall_score,average='macro')
            self.scorer = make_scorer(roc_auc_score)
        else:
            self.scorer = make_scorer(f1_score)#recall_score,average='macro')
        if 'augment_grp' in data.columns:
            train_test_splitter = sklearn.model_selection.GroupKFold(n_splits=4)
            for train_idxs, test_idxs in train_test_splitter.split(data, label, data['augment_grp']):
                x_train = data.iloc[train_idxs,:]
                x_test = data.iloc[test_idxs,:]
                y_train = label.iloc[train_idxs]
                y_test = label.iloc[test_idxs]
                x_train.pop('augment_grp')
                x_test.pop('augment_grp')
        else:
            x_train,x_test,y_train,y_test = train_test_split(data,label,train_size=0.75,stratify=label)
        self.features = x_train.columns
        [setattr(self,name,cur_attr) for name,cur_attr in zip(['x_train','x_test','y_train','y_test','label'],[x_train,x_test,y_train,y_test,label])]
        return x_train,x_test,y_train,y_test
    

    def objective(self,params):
        self.model_init(params)  
        cv_score = cross_val_score(self.model, self.x_train, self.y_train, cv=5,scoring=self.scorer)
        return np.mean(cv_score)
   
    def fit(self):
        self.model.fit(self.x_train,self.y_train)
        self.evalute()
    def plot(self):
        if self.model_type=='DT':
            self.plot_one_tree(self.model)
        elif self.model_type=='RF':
            pre_save_prefix = self.save_prefix
            for i in range(5):
                self.save_prefix = 'Tree'+str(i)+'_'
                self.plot_one_tree(self.model.estimators_[i])
            self.save_prefix = pre_save_prefix
        ax = plt.gca()
        #self.roc_disp = RocCurveDisplay.from_estimator(self.model, self.x_test, self.y_test, ax=ax, alpha=0.8)
        self.roc_disp = sklearn.metrics.plot_roc_curve(self.model, self.x_test, self.y_test, ax=ax, alpha=0.8)
        explainer = shap.Explainer(self.model)
        plt.show()
        shap_values = explainer(self.x_test)
        
        # visualize the first prediction's explanation
        if self.model_type=='cat':
            shap.plots.waterfall(shap_values[0])
            plt.show()
        #shap.plots.force(shap_values[:500])
            plt.show()
            shap.summary_plot(shap_values, self.x_test)
        #shap_interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(self.x_test.iloc[:200,:])
        #shap.summary_plot(shap_interaction_values, self.x_test.iloc[:200,:])
    def plot_one_tree(self,model_for_plot):
        dotfile= StringIO()
        sklearn.tree.export_graphviz(
              model_for_plot,  
              out_file        = dotfile,
              feature_names   = self.features, 
              class_names     = ['no', 'Yes'], 
              filled          = True,
              proportion=True,
              rounded         = True
          )
        graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
        graph.write_png(self.save_prefix+'_'+self.model_type+'_decisiontree.png')
      
        plt.figure(figsize=(20,20))
        #img = graphviz.Source(graph_from_dot_data(dotfile.getvalue()))
        img=pltimg.imread(self.save_prefix+'_'+self.model_type+'_decisiontree.png')
        plt.imshow(img)
        '''
        plt.figure(figsize=(30,30))
        plot_tree(model_for_plot,feature_names   = self.features, 
                      class_names= ['no', 'Readmission'] ,impurity=True,
                      proportion=True,filled= True,rounded= True,fontsize=12)'''
        plt.show()
        
    def evalute(self):
        if self.model_type!='cat':
            train_samples_weights = compute_sample_weight('balanced', self.y_train)
            test_samples_weights = compute_sample_weight('balanced', self.y_test)
            train_weighted_acc = self.model.score(self.x_train,self.y_train, sample_weight=train_samples_weights)
            test_weighted_acc = self.model.score(self.x_test,self.y_test, sample_weight=test_samples_weights)
            print(f'acc on train:{train_weighted_acc} test:{test_weighted_acc}')
        '''
        cm = confusion_matrix(self.y_test, self.model.predict(self.x_test))
      return {'tn': cm[0, 0], 'fp': cm[0, 1],
              'fn': cm[1, 0], 'tp': cm[1, 1]}
        '''
        plt.show()
        labels=['No', str(self.label_name)]
        print(classification_report(self.y_test, self.model.predict(self.x_test), target_names=labels))
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='true')
        plt.title(self.save_prefix+'_'+self.model_type+' norm over true')
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='pred')
        plt.title(self.save_prefix+'_'+self.model_type + ' norm over pred')
        plt.savefig(self.model_type + ' norm over pred.png')
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='all')
        plt.title(self.save_prefix+'_'+self.model_type + ' norm overall')
        plt.show()
        self.perm_imp()
    def perm_imp(self):
        from sklearn.inspection import permutation_importance
        r = permutation_importance(self.model, self.x_train, self.y_train,n_repeats=2,random_state=0)
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"{self.features[i]:<8}"
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")
    def load(self,model_path=False):
            if model_path:
                filename = model_path
            else:
                filename = self.save_prefix+'_'+self.model_type +'model.pkl'
            with open(filename, 'rb') as handle:
                self.model = pickle.load(handle)
                

start_time = time.time()
model_path = 'full_dataset_catmodel.pkl'
tree2 = clf(label_name=90,model_type='cat',dataset_filename='over_under_aug_dataset_cat.pkl',model_path = model_path)
tree2.multi_obejctive=False
x_train,x_test,y_train,y_test = tree2.data_read()
data_load_time=time.time()

tree2.evalute()

fit_time=time.time()
tree2.plot()
roc_plot_list=[]
roc_plot_list.append(tree2.roc_disp)
end = time.time()

start_time = time.time()
model_path = 'overunder_dataset_catmodel.pkl'
tree2 = clf(label_name=90,model_type='cat',dataset_filename='over_under_aug_dataset_cat.pkl',model_path = model_path)
tree2.multi_obejctive=False
x_train,x_test,y_train,y_test = tree2.data_read()
data_load_time=time.time()

tree2.evalute()

fit_time=time.time()
tree2.plot()
roc_plot_list.append(tree2.roc_disp)
end = time.time()
print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
print_time('Data load',start_time,data_load_time)
print_time('Fit',data_load_time,fit_time)
print_time('Overall time:',start_time,end)


'''
ax = plt.gca()
for i in roc_plot_list:
    i.plot(ax=ax, alpha=0.8)
'''
