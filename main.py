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
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.image as pltimg
from io import StringIO
from sklearn.inspection import permutation_importance
from functools import partial
import time
import pickle
import optuna
import plotly
import plotly.io as pio
pio.renderers.default='browser'
import shap
from catboost import CatBoostClassifier
from pdpbox import pdp, get_dataset, info_plots
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler

#conda install catboost shap plotly optuna pickle
class clf():
    def __init__(self,label_name, model_type='DT',filename='preproccesed_dataset.pkl',
                 save_prefix='',load_exist_model=False,models_list=[],
                 val_split_method='last_years',sampling_rates=[1]):
        self.model_type = model_type
        self.filename = filename
        self.label_name = label_name
        self.save_prefix = save_prefix
        self.multi_obejctive = False
        self.load_exist_model = load_exist_model
        self.models_list = models_list
        self.resampling_options_saved = False
        self.data_rescaled = False
        self.resampling_rates = sampling_rates#[1,0.6,0.4,0]
        self.val_split_method = val_split_method
        if load_exist_model:
            self.load(load_exist_model)
        else:
            self.model_init()
        
 
    def data_read(self,short=False,restrict_samples_n=False,split_train_test=True):
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
            #data.pop('Blood Type')
            for feat in self.cat_feats:
                if feat in data.columns and data[feat].dtypes == 'float64':
                    data[feat] =data[feat].astype('int64')
            self.cat_feats = pd.read_excel(io='categorical features.xlsx').columns 
            self.cat_feats = self.cat_feats[self.cat_feats.isin(data.columns)].to_numpy()
            self.non_cat_feats = data.columns[data.columns.isin(self.cat_feats)].to_numpy()
            self.cat_feats = self.cat_feats[self.cat_feats!=self.label_name]
            self.model.cat_features = self.cat_feats.tolist()
        data = data.query('`Induction of labor`>=0 and Parity==1 and `Past_preterm birth`==0')
        #after_corona_start = data['Date of delivery2']>pd.Timestamp(year=2020,month=7,day=1)
        #before_corona_end = data['Date of delivery2']<pd.Timestamp(year=2021,month=2,day=1)
        in_corona_times = np.logical_and(data['Date of delivery2']>pd.Timestamp(year=2020,month=7,day=1),data['Date of delivery2']<pd.Timestamp(year=2021,month=2,day=1))
        data = data.loc[~in_corona_times]
        print('delete corona time cases!')
        cols_with_0_var = data.var().index[data.var()==0].tolist()
        data = data.drop(cols_with_0_var,axis=1)
        label = data['label']
        data.pop('label')
        self.scorer=self.get_scorer(label)
        
        if split_train_test:
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
                if self.val_split_method == 'random':
                    #random split
                    data.pop('year')
                    data.pop('Date of delivery2')
                    x_train,x_test,y_train,y_test = train_test_split(data,label,train_size=0.75,stratify=label)
                elif self.val_split_method == 'last_years':
                    #last years split
                    train_idx = data.year.isin(list(range(2005,2018)))
                    val_idx = data.year.isin([2019,2020,2021])
                    data.pop('year')
                    data.pop('Date of delivery2')
                    x_train,y_train,x_test,y_test = data.loc[train_idx,:],label[train_idx], data.loc[val_idx,:],label[val_idx]
            if not self.is_balanced and not self.load_exist_model:
                #TODO: resampling only for trainable model (and not from model obtained in model_inspect.py script)
                x_train_wo_resampling, y_train_wo_resampling = x_train, y_train
                x_train,y_train = self.resample(x_train,y_train)
            self.features = x_train.columns
            [setattr(self,name,cur_attr) for name,cur_attr in zip(['x_train','x_test','y_train','y_test','x_train_wo_resampling','y_train_wo_resampling','label'],[x_train,x_test,y_train,y_test,x_train_wo_resampling,y_train_wo_resampling,label])]
            return x_train,x_test,y_train,y_test
        else:
            [setattr(self,name,cur_attr) for name,cur_attr in zip(['x','y','data','label'],[data,label,data,label])]
            return data,label
    def data_init(self,x_train,x_test,y_train,y_test,label,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=False):
        if x_train_dict:
            self.x_train_dict=x_train_dict
            self.resampling_options_saved=True
        self.features = x_train.columns
        self.scorer=self.get_scorer(label)
        [setattr(self,name,cur_attr) for name,cur_attr in zip(['x_train','x_test','y_train','y_test','x_train_wo_resampling','y_train_wo_resampling','label'],[x_train,x_test,y_train,y_test,x_train_wo_resampling,y_train_wo_resampling,label])]
        return x_train,x_test,y_train,y_test
    def get_scorer(self,label):
        print('Imbalance {:.2f}% ({:.0f}) / {:.2f}% ({:.0f}) - Label:{} / NO'.format(
            label.mean()*100,
            label.sum(),
            ((1-label.mean())*100),
            len(label)-label.sum(),
            self.label_name))
        self.imbalance_rate = label.sum()/len(label)
        self.is_balanced = np.abs(self.imbalance_rate-0.5)<0.1
        if self.is_balanced:
            #self.scorer = make_scorer(balanced_accuracy_score,greater_is_better=True)#recall_score,average='macro')
            #self.scorer = make_scorer(recall_score,average='macro')
            return make_scorer(balanced_accuracy_score,greater_is_better=True)#make_scorer(roc_auc_score)
        else:
            #self.scorer = make_scorer(recall_score,average='macro')
            #self.scorer = make_scorer(f1_score)#recall_score,average='macro')
            return make_scorer(balanced_accuracy_score,greater_is_better=True) #make_scorer(roc_auc_score)
    def resample(self,x,y,undersample_percent=0.5,from_saved_resampling=False):
        if from_saved_resampling and not self.resampling_options_saved:
            self.x_train_dict={a:self.resample(x,y,undersample_percent=a,from_saved_resampling=False) for a in self.resampling_rates}
            self.resampling_options_saved = True
            return self.x_train_dict[undersample_percent]
        elif from_saved_resampling and self.resampling_options_saved:
            return self.x_train_dict[undersample_percent]
        tl = TomekLinks(sampling_strategy='majority')
        x_tl, y_tl = tl.fit_resample(x, y)
        x_tl, y_tl = x ,y 
        print('Original(train subset) dataset shape', len(y))
        print('TomekLinks undersample(train subset)  dataset shape', len(y_tl))
        # sampling_strategy if float the function resample the major so the N(minor)/N(major after resmapling)=float
        if undersample_percent>0:
            rus = RandomUnderSampler(sampling_strategy=undersample_percent, replacement=False)# fit predictor and target variable
            x_rus, y_rus = rus.fit_resample(x_tl, y_tl)
            print(f'Random undersample {undersample_percent} (train subset) dataset shape{len(y_rus)}')
        else:
            x_rus, y_rus =  x_tl, y_tl
            print(f'NO undersampling (train subset) dataset shape{len(y_rus)}') 
        from imblearn.over_sampling import SMOTENC
        if self.model_type=='cat':
            idx_of_cat_feats = [i for i,feat in enumerate(x.columns) if feat in self.cat_feats]
            smote = SMOTENC(categorical_features=idx_of_cat_feats)
        else:
            bin_feat = [i for i,feat in enumerate(x.columns) if len(x[feat].value_counts())<4]
            smote = SMOTENC(categorical_features=bin_feat)
        
        x_smote, y_smote = smote.fit_resample(x_rus, y_rus)
        if 'Hemoglobin drop, gram/dl' in x_smote.columns: #Hemoglobin drop, gram/dl is the only feat with unrounded values
            cont_var = x_smote['Hemoglobin drop, gram/dl ']
            x_smote = x_smote.round()
            x_smote['Hemoglobin drop, gram/dl']=cont_var
        else:
            x_smote['First_vaginaltest']*=2 #enable 0.5 values
            x_smote = x_smote.round()
            x_smote['First_vaginaltest']/=2
        print('Dataset(train subset) shape after SMOTE upsample', len(y_smote))
        print('Train Imbalance after resampling {:.2f}% ({:.0f}) / {:.2f}% ({:.0f}) - Label:{} / NO'.format(
            y_smote.mean()*100,
            y_smote.sum(),
            ((1-y_smote.mean())*100),
            len(y_smote)-y_smote.sum(),
            self.label_name))
        return x_smote,y_smote
    def model_init(self,params={}):
        if params=={}:   
            if self.model_type=='DT':
                self.model = DecisionTreeClassifier(class_weight='balanced',**params)
            elif self.model_type=='RF':
                params ={'max_samples': 0.9968258002383812,
                 #                'min_samples_split': 10,
                                 'max_depth': 8,
                                 'max_features': 0.0037916994025758433}
                self.model = RandomForestClassifier(n_estimators=100, criterion='gini',min_samples_leaf=10,
                                     max_leaf_nodes=None,
                                      bootstrap=True, n_jobs=4, random_state=None, verbose= 0,class_weight='balanced_subsample',**params)    
            elif self.model_type=='cat':
               #0.84 {'learning_rate': 0.4394793483155559, 'iterations': 25}. Best is trial 11 with value: 0.849023425351254.
                       # self.cat_feats = pd.read_excel(io='categorical features.xlsx').columns 
                        #self.cat_feats = self.cat_feats[self.cat_feats.isin(self.data.columns)].to_numpy()
                      #  self.cat_feats = self.cat_feats[self.cat_feats!=self.label_name]
                        self.model = CatBoostClassifier(
                            iterations=10,
                             verbose=False,
                             nan_mode = 'Forbidden',
                             #,learning_rate=0.1
                             #model_size_reg
                             # class_names=['no', 'Readmission'],
                             auto_class_weights='Balanced',
                             one_hot_max_size=3,
                             #cat_features = self.cat_feats # the categorical features hasn't found yet
                        )

            elif self.model_type=='logi':
                self.model=LogisticRegression()
            elif self.model_type=='stack':
                  [check_is_fitted(cur_model.model) for cur_model in self.models_list]
                  self.model = stacked_clf(self.models_list)
        else:
            if 'resampling_rate' in params.keys():
                self.x_train,self.y_train = self.resample(self.x_train_wo_resampling, self.y_train_wo_resampling,undersample_percent=params['resampling_rate'],from_saved_resampling=True)
                params = {k: params[k] for k in set(list(params.keys())) - set(['resampling_rate'])}
            self.model.set_params(**params)
            
    def objective(self,params):
        self.model_init(params)  
        cv_score = cross_val_score(self.model, self.x_train, self.y_train, cv=5,scoring=self.scorer)
        return np.mean(cv_score)
    def optuna_objective(self,trial):
        if self.model_type=='DT':
            params ={
            #'min_samples_split' : trial.suggest_int(name="min_samples_split", low=3, high=14),
            'max_depth' : trial.suggest_int(name="max_depth", low=4, high=15),
            #'ccp_alpha':trial.suggest_float(name='ccp_alpha',low=0,high=1),
            'min_samples_leaf' : trial.suggest_float(name='min_samples_leaf',low=0.02,high=0.2),
            'max_leaf_nodes':trial.suggest_int(name='max_leaf_nodes',low=20,high=720,step=50),
            'splitter':trial.suggest_categorical(name='splitter', choices=['best', 'random'])
            }
        elif self.model_type=='RF':
            params ={
                'n_estimators':trial.suggest_int(name="n_estimators", low=10, high=350),
            'max_samples' : trial.suggest_float(name="max_samples", low=0.2, high=1),
            'min_samples_leaf' : trial.suggest_float(name='min_samples_leaf',low=0.02,high=0.3),
            'max_depth' : trial.suggest_int(name="max_depth", low=2, high=15),
            'max_features' : trial.suggest_float(name="max_features", low=0, high=1)
            }
        elif self.model_type=='cat':
            params = {'learning_rate':trial.suggest_loguniform(name='learning_rate',low=1e-6,high=1), 
                      'iterations':trial.suggest_int(name='iterations',low=5,high=50),
                      'depth':trial.suggest_int(name='depth',low=2,high=8),
                      'l2_leaf_reg':trial.suggest_float(name='l2_leaf_reg',low=0,high=10)
                      #'max_leaves':trial.suggest_int(name='max_leaves',low=5,high=500,step=10)
                      }
        elif self.model_type=='logi':
            if not self.data_rescaled:
                    self.rescale()
            params={'C':trial.suggest_float(name='C',low=0.001,high=8),
                    'max_iter':trial.suggest_int(name='max_iter',low=20,high=200,step=10),
                    'penalty':trial.suggest_categorical(name='penalty', choices=['l1', 'l2'])
                    }
        #params['class_weight']=trial.suggest_categorical(name='class_weight', choices=['balanced', None])
        params['resampling_rate'] = trial.suggest_categorical(name='resampling_rate',choices=self.resampling_rates)
        acc  = self.objective(params)

        if self.multi_obejctive:
            acc += params['min_samples_leaf']/1000
        return acc
    def rescale(self):
        from sklearn import preprocessing
        #TODO: scale only non cat features
        self.model.scaler = preprocessing.StandardScaler().fit(self.x_train)#[self.non_cat_feats])
        self.x_train = self.model.scaler.transform(self.x_train)#[self.non_cat_feats])
        self.x_test = self.model.scaler.transform(self.x_test)#[self.non_cat_feats])
        self.data_rescaled = True
    def hpo(self,time_in_min=5):
        study = optuna.create_study(study_name=self.model_type+' study',direction='maximize')
        print(f'Scorer func:{self.scorer._score_func.__name__}')
        study.optimize(self.optuna_objective, timeout=time_in_min*60)
        print(f'Scorer f:{self.scorer._score_func.__name__}')
        #optuna.visualization.plot_intermediate_values(study)
        ad = optuna.visualization.plot_slice(study)
        ad.show()
        if study.best_trial.number>1: # if there is only one trial or there is zero variance, param importance is unavliable
            print('feat importance:')
            print(optuna.importance.get_param_importances(study))
        print(f'\nbest param:{study.best_params}')
        optuna.visualization.plot_slice(study)
        plt.show()
        self.model_init(params=study.best_params)
        self.model.fit(self.x_train,self.y_train)#,plot=True
        self.study = study
        self.external_validation()
        self.p_dist_from_true_label()
        return study.best_params, study
    def fit(self):
        if self.model_type=='cat':
            #TODO: check if there is meaning to pass the `cat_features=self.cat_feats`
            #self.model.fit(self.x_train,self.y_train,cat_features=self.cat_feats)#,text_features='Blood type')
            self.model.fit(self.x_train,self.y_train)
        else:
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
        self.roc_disp = RocCurveDisplay.from_estimator(self.model, self.x_test, self.y_test, ax=ax, alpha=0.8)
        #self.roc_disp = sklearn.metrics.plot_roc_curve(self.model, self.x_test, self.y_test, ax=ax, alpha=0.8)
        plt.show()
        
        # visualize the first prediction's explanation
        if self.model_type=='cat':
            self.shap_init()
            shap.plots.waterfall(self.shap_values[0])
            plt.show()
        #shap.plots.force(shap_values[:500])
            plt.show()
            self.shap_plot(plot_type='summary_plot')
            self.shap_plot(plot_type='dependence',dependence_plot_feat=[0,1,2,4,5],interaction_index='auto')
            #self.shap_plot(plot_type='force')
        if self.model_type=='logi':
            for feat, coef in zip(self.features,self.model.coef_[0]):
                rounded_coef = round(coef,3)
                if rounded_coef>0.001:
                    print(f'{feat}:{rounded_coef}')
        #shap.dependence_plot(1, shap_values, self.x_test,feature_names=self.x_test.columns)
        #shap_interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(self.x_test.iloc[:200,:])
        #shap.summary_plot(shap_interaction_values, self.x_test.iloc[:200,:])
    def p_dist_from_true_label(self):
        pos_idx = np.where(self.y_test==1)[0]
        neg_idx = np.where(self.y_test==0)[0]
        if isinstance(self.x_test,pd.DataFrame):
            test_pos = self.x_test.iloc[pos_idx,:]
            test_neg = self.x_test.iloc[neg_idx,:]
            
        elif isinstance(self.x_test,np.ndarray):
            test_pos = self.x_test[pos_idx,:]
            test_neg = self.x_test[neg_idx,:]
        neg_pred = self.model.predict_proba(test_neg)[:,1]
        pos_pred = self.model.predict_proba(test_pos)[:,1]
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),dpi=500)
        fig.suptitle('Distance between prediction and actual values(0/1) '+self.model_type)
        ax1.hist(1-pos_pred,stacked=True)
        ax1.set_title('Yes. μ = '+str(np.mean(1-pos_pred)))
        ax1.set_xlabel('1- risk probability')
        ax1.set_ylabel('Cases')
        ax2.hist(neg_pred,stacked=True)
        ax2.set_title('No. μ = '+str(np.mean(neg_pred)))
        ax2.set_xlabel('probability risk')
        ax2.set_ylabel('Cases')
        plt.show()
    def plot_one_tree(self,model_for_plot):
        dotfile= StringIO()
        sklearn.tree.export_graphviz(
              model_for_plot,  
              out_file        = dotfile,
              feature_names   = self.features, 
              class_names     = ['no', self.label_name], 
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
                      class_names= ['no', self.label_name] ,impurity=True,
                      proportion=True,filled= True,rounded= True,fontsize=12)'''
        plt.show()
        
    def evalute(self,y_test=None,x_test=None,title=''):
        if y_test is None:
            y_test=self.y_test
            x_test=self.x_test
        if self.model_type!='cat':
            train_samples_weights = compute_sample_weight('balanced', self.y_train)
            test_samples_weights = compute_sample_weight('balanced', self.y_test)
            train_weighted_acc = self.model.score(self.x_train,self.y_train, sample_weight=train_samples_weights)
            test_weighted_acc = self.model.score(self.x_test,self.y_test, sample_weight=test_samples_weights)
            print(f'acc on train:{train_weighted_acc} test:{test_weighted_acc}')
        '''
        cm = confusion_matrix(y_test, self.model.predict(x_test))
      return {'tn': cm[0, 0], 'fp': cm[0, 1],
              'fn': cm[1, 0], 'tp': cm[1, 1]}
        '''
        plt.show()
        labels=['No', str(self.label_name) ]
        print(classification_report(y_test, self.model.predict(x_test), target_names=labels))
        plot_confusion_matrix(self.model,x_test,y_test,display_labels=labels,normalize='true')
        plt.title(title+' '+self.save_prefix+'_'+self.model_type+' norm over true')
        plot_confusion_matrix(self.model,x_test,y_test,display_labels=labels,normalize='pred')
        plt.title(title+' '+self.save_prefix+'_'+self.model_type + ' norm over pred')
        plt.savefig(title+' '+self.model_type + ' norm over pred.png')
        plot_confusion_matrix(self.model,x_test,y_test,display_labels=labels,normalize='all')
        plt.title(title+' '+self.save_prefix+'_'+self.model_type + ' norm overall')
        plt.show()
    def shap_init(self):
        self.explainer = shap.TreeExplainer(self.model)#, data=self.x_train)
        #self.explainer = shap.Explainer(self.model,feature_names=self.features)
        #TODO: all shap base values, are the same - print(shap_values.base_values). i think it is wrong, and i need to pass alterantive base values. 
        self.shap_values = self.explainer(self.x_test)
        self.shap_train_values = self.explainer(self.x_train)
        self.feat_importance = np.argsort(-np.abs(self.shap_values.values.mean(axis=0)))
        return self.shap_values
    def shap_plot(self,plot_type='summary',dependence_plot_feat=False,interaction_index='auto',specific_sample=False):
        if not hasattr(self, 'shap_values'):
            self.shap_init()
        if (plot_type=='summary' or plot_type=='summary_plot') and not specific_sample:
            shap.summary_plot(self.shap_values, self.x_test)
        elif plot_type=='dependence':
            
            if not dependence_plot_feat:
                dependence_plot_feat = 0 #self.features[self.feat_importance[0]]
            #elif dependence_plot_feat.isnumeric():
            #    dependence_plot_feat = self.features[self.feat_importance[dependence_plot_feat]]
            #or:
            if type(dependence_plot_feat)==str and dependence_plot_feat in self.features:
                shap.dependence_plot(dependence_plot_feat, self.shap_values.values, self.x_test,interaction_index=interaction_index,
                             xmin='percentile(1)',xmax='percentile(99)',x_jitter=1)
            elif dependence_plot_feat and interaction_index != 'auto':
                for feat in dependence_plot_feat:
                    for interact_feat in interaction_index:
                        shap.dependence_plot('rank('+str(feat)+')', self.shap_values.values, self.x_test,interaction_index=str(interact_feat),
                                     xmin='percentile(1)',xmax='percentile(99)',x_jitter=1)
            else:
                for feat in dependence_plot_feat:
                        shap.dependence_plot('rank('+str(feat)+')', self.shap_values.values, self.x_test,interaction_index='auto',
                                     xmin='percentile(1)',xmax='percentile(99)',x_jitter=1)
      
            # in third case, the feature can be an argmuent sent by the caller
                
            #shap.dependence_plot(dependence_plot_feat, self.shap_values, self.x_test,feature_names=self.x_test.columns)
        elif plot_type=='force':
            n_trials=min([2000,len(self.x_test)])
            trials_sampled_idx = np.random.choice(range(len(tree2.x_test)),size=n_trials)
            shap.plots.force(self.shap_values_test[trials_sampled_idx])
        elif specific_sample:
            shap.plots.waterfall(self.shap_values[specific_sample])
        plt.show()
    def save(self):
        self.model.train_idx=self.y_train.index
        self.model.test_idx=self.y_test.index
        filename = self.save_prefix+'_'+self.model_type +'model.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,model_path=False):
            if model_path:
                filename = model_path
            else:
                filename = self.save_prefix+'_'+self.model_type +'model.pkl'
            with open(filename, 'rb') as handle:
                self.model = pickle.load(handle)
    def external_validation(self,external_val_filename=False):
        if not external_val_filename:
            external_val_filename='BH_'+self.filename[3:]
        with open(external_val_filename, 'rb') as handle:
            external_data = pickle.load(handle)
        self.external_y_test = external_data['label']
        
        self.external_x_test = external_data[self.features].copy()
        if len(self.external_y_test)>0:
            if self.model_type=='logi':
                self.external_x_test = self.model.scaler.transform(self.external_x_test)
            self.evalute(y_test=self.external_y_test,x_test=self.external_x_test,title='External validity' )
            ax = plt.gca()
            #self.roc_disp = RocCurveDisplay.from_estimator(self.model, external_x_test, external_y_test, ax=ax, alpha=0.8)
            self.ext_roc_disp = sklearn.metrics.plot_roc_curve(self.model, self.external_x_test, self.external_y_test, ax=ax, alpha=0.8)
            plt.show()
        else:
            print('There is no external validity dataset (0 records)')
###
class stacked_clf(BaseEstimator, ClassifierMixin):

   def __init__(self, models_list,clf_vote_type=''):
        if clf_vote_type=='':
            if len(models_list)%2:
                clf_vote_type='proba_vote'
            else:
                clf_vote_type='bin_vote'
        self.models_list = models_list
        self.models_len = len(models_list)
        self.clf_vote_type = clf_vote_type
   def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if self.clf_vote_type=='logi_layer':
            self.logi_layer = LogisticRegression().fit(self.each_model_predict_proba(X)[:,:,1].T,y)
            models_coef = np.exp(self.logi_layer.coef_)/np.sum(np.exp(self.logi_layer.coef_))
        self.X_ = X
        self.y_ = y
        return self
   def predict(self, X):
        # Check is fit had been called    
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        if self.clf_vote_type == 'bin_vote':
        # vote by binary prediction 
           return self.each_model_predict(X).mean(axis=0).round()
        # other option - predict by predict_proba
        elif self.clf_vote_type == 'proba_vote':
            return np.argmax( self.predict_proba(X), axis=1)
        elif self.clf_vote_type == 'logi_layer':
            return self.logi_layer.predict(self.each_model_predict_proba(X))
   def predict_proba(self, X):
       # Check is fit had been called    
       check_is_fitted(self)
       # Input validation
       X = check_array(X)
       if self.clf_vote_type=='logi_layer':
            self.X_ = self.logi_layer.predict_proba(self.each_model_predict_proba(X))
       else:
           self.X_ = self.each_model_predict_proba(X).mean(axis=0)
       return self.X_
   def each_model_predict_proba(self,X):
       model_pred = np.zeros((self.models_len,len(X),2))
       for i,cur_model in enumerate(self.models_list):
           model_pred[i,:,:] = cur_model.model.predict_proba(X)
       return model_pred
   def each_model_predict(self,X):
       return np.argmax(self.each_model_predict_proba(X),axis=2)
   def plot(self,X,y):
        corr_scores = self.predict(X)==y
        model_pred = self.each_model_predict_proba(X)[:,:,1]
        corr_pred_each = model_pred.round()==np.tile(y,(self.models_len ,1))
        corr_models = corr_pred_each.sum(axis=0)
        pd.DataFrame(corr_models[corr_scores==False]).value_counts()
        print(f'Stacked models obtained{corr_scores.sum()*100/len(corr_scores)}% acc')
        ax = plt.gca()
        RocCurveDisplay.from_estimator(self, X, y, ax=ax, alpha=0.8)
        plt.show()

        
####################################################################################
####################################################################################
####################################################################################
if __name__ == "__main__":              
    time_for_each_model_min=5
    label_name = 'In labor cesarean'
    #tree = clf(label_name=90,model_type='DT',save_prefix='full_dataset')
    #full_data,full_label = tree.data_read(split_train_test=False)
    roc_plot_list=[]
    #ext_roc_plot_list=[]
    start_time = time.time()
    
    start_time = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    '''
    tree5 = clf(label_name=label_name,model_type='cat',filename='SZ_preproccesed_dataset+vaginal_test_cat.pkl',save_prefix='full_dataset')
    
    tree5.multi_obejctive=False
    x_train,x_test,y_train,y_test = tree5.data_read()
    print(f'data len{len(y_train)+len(y_test)}')
    data_load_time=time.time()
    tree5.hpo(time_in_min=time_for_each_model_min)
    tree5.evalute()
    tree5.save()
    fit_time=time.time()
    tree5.plot()
    roc_plot_list.append(tree5.roc_disp)
    '''
    tree2 = clf(label_name=label_name,model_type='cat',filename='SZ_preproccesed_dataset_cat.pkl',
                save_prefix='full_dataset')
    
    tree2.multi_obejctive=False
    x_train,x_test,y_train,y_test = tree2.data_read()
    print(f'data len{len(y_train)+len(y_test)}')
    data_load_time=time.time()
    tree2.hpo(time_in_min=time_for_each_model_min)
    tree2.evalute()
    tree2.save()
    fit_time=time.time()
    tree2.plot()
    roc_plot_list.append(tree2.roc_disp)
  
    x_train_wo_resampling,y_train_wo_resampling,x_train_dict = tree2.x_train_wo_resampling, tree2.y_train_wo_resampling, tree2.x_train_dict
    start_time = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))

    tree5 = clf(label_name=label_name,model_type='cat',filename='SZ_preproccesed_dataset+temp_cat.pkl',save_prefix='cat+temp')
    x_train1,x_test1,y_train1,y_test1 = tree5.data_read()
    data_load_time=time.time()
    tree5.hpo(time_in_min=time_for_each_model_min)
    tree5.evalute(title='+temp')
    tree5.save()
    fit_time=time.time()
    tree5.plot()
    roc_plot_list.append(tree5.roc_disp)
    #ext_roc_plot_list.append(tree4.ext_roc_disp)
    '''
    tree4 = clf(label_name=label_name,model_type='logi',filename='SZ_preproccesed_dataset_cat.pkl',save_prefix='overunder_dataset')
    tree4.data_init(x_train,x_test,y_train,y_test,y_test,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=x_train_dict)
    data_load_time=time.time()
    tree4.hpo(time_in_min=time_for_each_model_min)
    tree4.evalute()
    tree4.save()
    fit_time=time.time()
    tree4.plot()
    roc_plot_list.append(tree4.roc_disp)
    #ext_roc_plot_list.append(tree4.ext_roc_disp)
    '''
    start_time = time.time()
    tree3 = clf(label_name=label_name,model_type='DT',filename='SZ_preproccesed_dataset_cat.pkl',save_prefix='overunder_dataset')
    tree3.data_init(x_train,x_test,y_train,y_test,y_test,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=x_train_dict)
    data_load_time=time.time()
    tree3.hpo(time_in_min=time_for_each_model_min)
    tree3.evalute()
    tree3.save()
    fit_time=time.time()
    tree3.plot()
    roc_plot_list.append(tree3.roc_disp)
    #ext_roc_plot_list.append(tree3.ext_roc_disp)
    
    end = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    
    
    start_time = time.time()
    tree1 = clf(label_name=label_name,model_type='RF',filename='SZ_preproccesed_dataset_cat.pkl',save_prefix='overunder_dataset')
    tree1.data_init(x_train,x_test,y_train,y_test,y_test,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=x_train_dict)
    data_load_time=time.time()
    tree1.hpo(time_in_min=time_for_each_model_min)
    tree1.evalute()
    tree1.save()
    fit_time=time.time()
    
    tree1.plot()
    roc_plot_list.append(tree1.roc_disp)
    #ext_roc_plot_list.append(tree1.ext_roc_disp)
    end = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    
    '''
    start_time = time.time()
    tree = clf(label_name=90,model_type='DT',save_prefix='full_dataset')
    x_train,x_test,y_train,y_test = tree.data_read()
    data_load_time=time.time()
    tree.hpo(time_in_min=45)
    tree.evalute()
    tree.save()
    fit_time=time.time()
    tree.plot()
    roc_plot_list.append(tree.roc_disp)
    end = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    '''
    
    '''
    start_time = time.time()
    tree = clf(label_name=90,model_type='RF',save_prefix='full_dataset')
    x_train,x_test,y_train,y_test = tree.data_read()
    data_load_time=time.time()
    tree.hpo(time_in_min=45)
    tree.evalute()
    tree.save()
    fit_time=time.time()
    print(f'acc on train:{tree.model.score(x_train,y_train)} test:{tree.model.score(x_test,y_test)}')
    tree.plot()
    roc_plot_list.append(tree.roc_disp)
    end = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    '''
    stack =stacked_clf([tree2,tree3,tree1],clf_vote_type='bin_vote')
    stack.fit(x_train,y_train)
    stack.plot(x_test,y_test)
    
    stack_clf =clf(label_name=label_name,model_type='stack',filename='SZ_preproccesed_dataset_cat.pkl',save_prefix='full_dataset',models_list =[tree2,tree3,tree1])
    stack_clf.data_init(x_train,x_test,y_train,y_test,y_test,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=x_train_dict)
    stack_clf.model.fit(x_train,y_train)
    stack_clf.external_validation()
    stack_clf.plot()
    stack_clf.p_dist_from_true_label()
    roc_plot_list.append(stack_clf.roc_disp)
    stack_clf1 =clf(label_name=label_name,model_type='stack',filename='SZ_preproccesed_dataset_cat.pkl',save_prefix='full_dataset',models_list =[tree2,tree3])
    stack_clf1.data_init(x_train,x_test,y_train,y_test,y_test,x_train_wo_resampling,y_train_wo_resampling,x_train_dict=x_train_dict)
    stack_clf1.model.fit(x_train,y_train)
    stack_clf1.external_validation()
    stack_clf1.plot()
    stack_clf1.p_dist_from_true_label()
    roc_plot_list.append(stack_clf1.roc_disp)
    ax = plt.gca()
    for i in roc_plot_list:
        i.plot(ax=ax, alpha=0.8)
    plt.title('ROC plot')
    plt.show()
    if 'ext_roc_plot_list' in locals():
        ax = plt.gca()
        for i in ext_roc_plot_list:
            i.plot(ax=ax, alpha=0.8)
            plt.title('External validity ROC plot')
        plt.show()