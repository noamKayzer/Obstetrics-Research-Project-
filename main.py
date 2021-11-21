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
pio.renderers.default='svg'
import shap
from catboost import CatBoostClassifier
#conda install catboost shap plotly optuna pickle
class clf():
    def __init__(self,label_name, model_type='DT',filename='preproccesed_dataset.pkl',save_prefix=''):
        self.model_type = model_type
        self.filename = filename
        self.label_name = label_name
        self.model_init()
        
        self.save_prefix = save_prefix
        self.multi_obejctive = True
        
        
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
            ############
            ####### NEED TO RETURN BLOOD TYPE FOR LEARNING
            ###########
            data.pop('Blood Type')
            for feat in self.cat_feats:
                if feat in data.columns and data[feat].dtypes == 'float64':
                    data[feat] =data[feat].astype('int64')
            self.cat_feats = pd.read_excel(io='categorical features.xlsx').columns 
            self.cat_feats = self.cat_feats[self.cat_feats.isin(data.columns)].to_numpy()
            self.cat_feats = self.cat_feats[self.cat_feats!=self.label_name]
            self.model.cat_features = self.cat_feats.tolist()
        label = data['label']
        data.pop('label')
        print('Imbalance {:.2f}% ({:d}) / {:.2f}% ({:d}) - Label:{} / NO'.format(
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
            self.scorer = make_scorer(roc_auc_score)
        else:
            self.scorer = make_scorer(recall_score,average='macro')
            self.scorer = make_scorer(f1_score)#recall_score,average='macro')
            #self.scorer = make_scorer(roc_auc_score)
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
                x_train,x_test,y_train,y_test = train_test_split(data,label,train_size=0.75,stratify=label)
            if not self.is_balanced:
                x_train,y_train = self.resample(x_train,y_train)
            self.features = x_train.columns
            [setattr(self,name,cur_attr) for name,cur_attr in zip(['x_train','x_test','y_train','y_test','label'],[x_train,x_test,y_train,y_test,label])]
            return x_train,x_test,y_train,y_test
        else:
            [setattr(self,name,cur_attr) for name,cur_attr in zip(['x','y','data','label'],[data,label,data,label])]
            return data,label
    def resample(self,x,y):

        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks(sampling_strategy='majority')
        x_tl, y_tl = tl.fit_resample(x, y)
        print('Original(train subset) dataset shape', len(y))
        print('TomekLinks undersample(train subset)  dataset shape', len(y_tl))
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42, replacement=True)# fit predictor and target variable
        x_rus, y_rus = rus.fit_resample(x_tl, y_tl)
        print('Random undersample(train subset)  dataset shape', len(y_rus))
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(x_rus, y_rus)
        print('Dataset(train subset) shape after SMOTE upsample', len(y_smote))
        print('Train Imbalance after resampling {:.2f}% ({:d}) / {:.2f}% ({:d}) - Label:{} / NO'.format(
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
                '''
                if params=={}:
                    params = {'max_samples': 45603867150.2647407,
                              'min_samples_split': 5,
                              'max_depth': 8,
                              'max_features': 0.8542359383474984}
                '''

                params ={'max_samples': 0.9968258002383812,
                 #                'min_samples_split': 10,
                                 'max_depth': 8,
                                 'max_features': 0.0037916994025758433}
                self.model = RandomForestClassifier(n_estimators=150, criterion='gini',min_samples_leaf=10,
                                     max_leaf_nodes=None,
                                      bootstrap=True, n_jobs=-1, random_state=None, verbose= 0,class_weight='balanced_subsample',**params)    
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
                            # cat_features = self.cat_feats
                        )
                        '''                        
                      clf.fit(
                            X_train, y_train,
                            cat_features=cat_features,
                            eval_set=(X_val, y_val),
                        )'''
        else:
            self.model.set_params(**params)
            '''
            for feat,val in params.items():
                setattr(self.model,feat,val)
                '''
            
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
            'max_leaf_nodes':trial.suggest_int(name='max_leaf_nodes',low=50,high=700,step=50),
            'splitter':trial.suggest_categorical(name='splitter', choices=['best', 'random'])
            }
        elif self.model_type=='RF':
            params ={
            'max_samples' : trial.suggest_float(name="max_samples", low=0.2, high=1),
            'min_samples_leaf' : trial.suggest_float(name='min_samples_leaf',low=0.02,high=0.2),
            'max_depth' : trial.suggest_int(name="max_depth", low=2, high=15),
            'max_features' : trial.suggest_float(name="max_features", low=0, high=1)
            }
        elif self.model_type=='cat':
            params = {'learning_rate':trial.suggest_loguniform(name='learning_rate',low=1e-5,high=0.5), 
                      'iterations':trial.suggest_int(name='iterations',low=5,high=25),
                      #'max_depth':trial.suggest_int(name='max_depth',low=3,high=20),
                      #'max_leaves':trial.suggest_int(name='max_leaves',low=5,high=500,step=10)
                      }
        #params['class_weight']=trial.suggest_categorical(name='class_weight', choices=['balanced', None])
        acc  = self.objective(params)
        #print(acc)
        if self.multi_obejctive:
            acc += params['min_samples_leaf']/100
            #print(acc, params['min_samples_leaf'])
        return acc
    def hpo(self,time_in_min=5):
        study = optuna.create_study(study_name=self.model_type+' study',direction='maximize')
        print(f'Scorer func:{self.scorer._score_func.__name__}')
        study.optimize(self.optuna_objective, timeout=time_in_min*60)
        print(f'Scorer f:{self.scorer._score_func.__name__}')
        #optuna.visualization.plot_intermediate_values(study)
        optuna.visualization.plot_slice(study)
        if study.best_trial.number>1: # if there is only one trial or there is zero variance, param importance is unavliable
            print('feat importance:')
            print(optuna.importance.get_param_importances(study))
        print(f'\nbest param:{study.best_params}')
        self.model_init(params=study.best_params)
        self.model.fit(self.x_train,self.y_train)
        self.study = study
        return study.best_params, study
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
        labels=['No', str(self.label_name) ]
        print(classification_report(self.y_test, self.model.predict(self.x_test), target_names=labels))
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='true')
        plt.title(self.save_prefix+'_'+self.model_type+' norm over true')
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='pred')
        plt.title(self.save_prefix+'_'+self.model_type + ' norm over pred')
        plt.savefig(self.model_type + ' norm over pred.png')
        plot_confusion_matrix(self.model,self.x_test,self.y_test,display_labels=labels,normalize='all')
        plt.title(self.save_prefix+'_'+self.model_type + ' norm overall')
        plt.show()
    def save(self):
        filename = self.save_prefix+'_'+self.model_type +'model.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self):
            filename = self.save_prefix+'_'+self.model_type +'model.pkl'
            with open(filename, 'rb') as handle:
                self.model = pickle.load(handle)
                
#####################
if __name__ == "__main__":              
    time_for_each_model_min=30
    label_name = 'In labor cesarean'
    #tree = clf(label_name=90,model_type='DT',save_prefix='full_dataset')
    #full_data,full_label = tree.data_read(split_train_test=False)
    start_time = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    roc_plot_list=[]
    tree2 = clf(label_name=label_name,model_type='cat',filename='preproccesed_dataset_cat.pkl',save_prefix='full_dataset')
    
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
    #tree2.x_test = full_data
    #tree2.y_test = full_label
    #tree2.evalute()
    end = time.time()
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    
    
    start_time = time.time()
    tree3 = clf(label_name=label_name,model_type='DT',filename='preproccesed_dataset.pkl',save_prefix='overunder_dataset')
    x_train,x_test,y_train,y_test = tree3.data_read()
    data_load_time=time.time()
    tree3.hpo(time_in_min=time_for_each_model_min)
    tree3.evalute()
    tree3.save()
    fit_time=time.time()
    tree3.plot()
    roc_plot_list.append(tree3.roc_disp)
    
    end = time.time()
    print_time = lambda title, start,end: print('{} took {:.2f} sec'.format(title, end-start))
    print_time('Data load',start_time,data_load_time)
    print_time('Fit',data_load_time,fit_time)
    print_time('Overall time:',start_time,end)
    
    
    start_time = time.time()
    tree1 = clf(label_name=label_name,model_type='RF',filename='preproccesed_dataset.pkl',save_prefix='overunder_dataset')
    x_train,x_test,y_train,y_test = tree1.data_read()
    data_load_time=time.time()
    tree1.hpo(time_in_min=time_for_each_model_min)
    tree1.evalute()
    tree1.save()
    fit_time=time.time()
    
    tree1.plot()
    roc_plot_list.append(tree1.roc_disp)
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
    ax = plt.gca()
    for i in roc_plot_list:
        i.plot(ax=ax, alpha=0.8)