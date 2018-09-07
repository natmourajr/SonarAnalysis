"""
    This file contents some classification analysis functions
    """

import os
import time
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, auc
from sklearn import svm


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.models import load_model

import matplotlib.pyplot as plt

from Functions.ClassificationAnalysis import CnnClassificationAnalysis
from Functions.ConvolutionalNeuralNetworks import KerasModel
from Functions.CrossValidation import SonarRunsKFold
from Functions.NpUtils.DataTransformation import lofar2image
from Functions.NpUtils.Scores import trigger_score, recall_score, spIndex
from Functions.SystemIO import exists, mkdir

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('legend',**{'fontsize':15})
plt.rc('font', weight='bold')

from multiprocessing import Pool


class TrnInformation(object):
    def __init__(self, date='',
                 novelty_class=0,
                 n_folds=2,
                 norm='mapstd',
                 verbose=False,
                 gamma='auto',
                 kernel='rbf'
                 ):
        self.novelty_class = novelty_class
        self.n_folds = n_folds
        self.norm = norm
        self.verbose = verbose
        
        # train params
        self.gamma = gamma
        self.kernel = kernel
        
        self.CVO = None
        if date == '':
            self.date = time.strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.date = date
    
    def Print(self):
        print 'Class TrnInformation'
        print '\tDate %s'%(self.date)
        print '\tNovelty Class ', self.novelty_class
        print '\tNumber of Folds %i'%(self.n_folds)
        print '\tNormalization: %s'%(self.norm)
        if self.CVO is None:
            print '\tCVO is None'
        else:
            print '\tCVO is not None'
        if self.verbose:
            print '\tVerbose is True'
        else:
            print '\tVerbose is False'
        print '\tGamma value: ',self.gamma
        print '\tKernel: ',self.kernel

    def SplitTrainSet(self,trgt):
        # divide data in train and test for novelty detection
        CVO = cross_validation.StratifiedKFold(trgt[trgt!=self.novelty_class], self.n_folds)
        self.CVO = list(CVO)

    def save(self, path=''):
        print 'Save TrnInformation'
        if path == '':
            print 'No valid path...'
            return -1
        joblib.dump([self.date,self.n_folds, self.norm, self.CVO, self.gamma, self.kernel],path,compress=9)
        return 0

    def load(self, path):
        print 'Load TrnInformation'
        if not os.path.exists(path):
            print 'No valid path...'
            return -1
        [self.date,self.n_folds, self.norm, self.CVO, self.gamma, self.kernel] = joblib.load(path)

class NoveltyDetectionBaseClass(object):

    name = 'BaseClass'
    preproc_path = ''
    train_path = ''
    anal_path = ''
    
    date = None
    trn_info = None

    def __init__(self, name='BaseClass', preproc_path='', train_path='', anal_path=''):
        self.name = name
        self.preproc_path = preproc_path
        self.train_path = train_path
        self.anal_path = anal_path
        
        self.date = None
        self.trn_info = None

    def Print(self):
        print 'Class %s'%(self.name)
        print '\tPre-Proc. Data Path: ', self.preproc_path
        print '\tTraining Data Path: ', self.train_path
        print '\tAnalysis Data Path: ', self.anal_path

class SVMNoveltyDetection(NoveltyDetectionBaseClass):
    def preprocess(self, data, trgt, novelty_class=0, trn_info=None, fold=0):
        print 'SVMNoveltyDetection preprocess function'
    
        if self.trn_info is None and trn_info is None:
            # checar se existe o arquivo
            file_name = '%s/%s_%s_novelty_%i_trn_info.jbl'%(self.preproc_path,
                                                            self.trn_info.date,
                                                            self.name,novelty_class)
        
            if not os.path.exists(file_name):
                print 'No TrnInformation'
                return [None, None]
            else:
                self.trn_info.load(file_name)
        else:
            if not trn_info is None:
                self.trn_info = trn_info
                # checar se existe o arquivo
                file_name = '%s/%s_%s_novelty_%i_trn_info.jbl'%(self.preproc_path,
                                                                self.trn_info.date,
                                                                self.name, novelty_class)
                if not os.path.exists(file_name):
                    self.trn_info.save(file_name)
        
        
        if fold > self.trn_info.n_folds or fold < 0:
            print 'Invalid Fold...'
            return [None, None]
        
        if novelty_class > trgt.argmax(axis=1).max():
            print 'Invalid novelty class...'
            return [None, None]


        if self.trn_info.CVO is None:
            print 'No Cross Validation Obj'
            return -1

        train_id, test_id = self.trn_info.CVO[fold]

        # checar se existe o arquivo
        file_name = '%s/%s_%s_preproc_fold_%i_novelty_%i.jbl'%(self.preproc_path,
                                                               self.trn_info.date,
                                                               self.name,fold,novelty_class)
        if not os.path.exists(file_name):
            print 'SVMNoveltyDetection preprocess function: creating scaler for novelty %i - fold %i'%(novelty_class,
                                                                                                    fold)
            data_novelty = data[trgt[:,novelty_class]!=1,:]
            # normalize data based in train set
            if self.trn_info.norm == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(data_novelty[train_id,:])
            elif self.trn_info.norm == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data_novelty[train_id,:])
            elif self.trn_info.norm == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data_novelty[train_id,:])
            joblib.dump([scaler],file_name,compress=9)
        else:
            #print 'NeuralClassication preprocess function: loading scaler for fold %i'%(fold)
            [scaler] = joblib.load(file_name)

        data_proc = scaler.transform(data)
        
        # others preprocessing process
        
        return [data_proc,trgt]


    def train(self, data, trgt, novelty_class=0, trn_info=None, nu_value=0.1, fold=0):
        print 'SVMNoveltyDetection train function'
    
        if self.trn_info is None and trn_info is None:
            # checar se existe o arquivo
            file_name = '%s/%s_%s_novelty_%i_trn_info.jbl'%(self.preproc_path,
                                                            self.trn_info.date,
                                                            self.name,novelty_class)
            
            if not os.path.exists(file_name):
                print 'No TrnInformation'
                return [None]
            else:
                self.trn_info.load(file_name)
        else:
            if not trn_info is None:
                self.trn_info = trn_info
                # checar se existe o arquivo
                file_name = '%s/%s_%s_novelty_%i_trn_info.jbl'%(self.preproc_path,
                                                                self.trn_info.date,
                                                                self.name, novelty_class)
                if not os.path.exists(file_name):
                    self.trn_info.save(file_name)
                                                                    
                                                                    
        if fold > self.trn_info.n_folds or fold < 0:
            print 'Invalid Fold...'
            return [None]
                                                                                
        if novelty_class > trgt.argmax(axis=1).max():
            print 'Invalid novelty class...'
            return [None]

    
        if nu_value < 0.0 or nu_value >1.0:
            print 'Invalid nu value...'
            return None

        [data_preproc, trgt_preproc] = self.preprocess(data,trgt,
                                                       novelty_class=novelty_class,
                                                       trn_info=trn_info,
                                                       fold=fold)
        # checar se o arquivo existe
        nu_value_str = ("%.5f"%(nu_value)).replace('.','_')
        file_name = '%s/%s_%s_novelty_%i_train_fold_%i_nu_%s_model.jbl'%(self.train_path,
                                                                         self.trn_info.date,
                                                                         self.name,
                                                                         novelty_class,fold,
                                                                         nu_value_str)
        if not os.path.exists(file_name):
            print 'SVMNoveltyDetection: training classifiers for novelty %i - fold %i - nu %1.3f'%(novelty_class,
                                                                                                   fold, nu_value)
            classifiers = []
            data_novelty = data_preproc[trgt_preproc[:,novelty_class]!=1,:]
            train_id, test_id = self.trn_info.CVO[fold]
            
            for iclass in range(trgt.argmax(axis=1).max()+1):
                
                if iclass == novelty_class:
                    print 'Training novelty detector for class %i'%(iclass)
                    # novelty detector
                    classifiers.append(svm.OneClassSVM(nu=nu_value,
                                                       kernel=self.trn_info.kernel,
                                                       gamma=self.trn_info.gamma))
                    classifiers[iclass].fit(data_novelty[train_id,:])
                else:
                    print 'Training classifiers for class %i'%(iclass)
                    # classifiers
                    classifiers.append(svm.OneClassSVM(nu=nu_value,
                                                       kernel=self.trn_info.kernel,
                                                       gamma=self.trn_info.gamma))
                    classifiers[iclass].fit(data_preproc[trgt_preproc[:,iclass]!=1,:])
            joblib.dump(classifiers,file_name,compress=9)
        else:
            # load model
            classifiers=joblib.load(file_name)
        return classifiers


    def analysis(self, data, trgt, trn_info=None):
        print 'SVMNoveltyDetection analysis function'

    def analysis_output_hist(self, data, trgt, trn_info=None, nu_value=0.1, fold=0):
        print 'SVMNoveltyDetection analysis output hist function'
        
        # checar se a analise ja foi feita
        nu_value_str = ("%.5f"%(nu_value)).replace('.','_')
        file_name = '%s/%s_%s_fold_%i_nu_%s_novelty_output_hist.jbl'%(self.train_path,
                                                                      self.trn_info.date,
                                                                      self.name,fold,
                                                                      nu_value_str)
        output = None
        if not os.path.exists(file_name):
            output = np.zeros([data.shape[0],trgt.shape[1]])
            for novelty_class in range(trgt.shape[1]):
                classifiers = self.train(data,trgt,novelty_class=novelty_class,
                                         trn_info=trn_info,nu_value=nu_value,fold=fold)
                output[:,novelty_class] = classifiers[novelty_class].predict(data)
            joblib.dump(output,file_name,compress=9)
        else:
            output = joblib.load(file_name)


        fig, ax = plt.subplots(figsize=(15,15),nrows=trgt.shape[1], ncols=output.shape[1])
        m_colors = ['b', 'r', 'g', 'y']
        m_bins = np.round(np.linspace(-1.5, 1.5, 10),decimals=1)

        for i_target in range(trgt.shape[1]):
            for i_output in range(output.shape[1]):
                subplot_id = output.shape[1]*i_target+i_output
                # alvos max esparsos
                m_pts = output[np.argmax(trgt,axis=1)==i_target,i_output]
                n, bins, patches = ax[i_target,i_output].hist(m_pts,bins=m_bins,
                                                              fc=m_colors[i_target],
                                                              alpha=0.8, normed=True)
                                                              
                ax[i_target,i_output].set_xticks(m_bins)
                ax[i_target,i_output].set_xticklabels(m_bins,rotation=45)
            
            
                if i_output == 0:
                    ax[i_target,i_output].set_ylabel('Novelty %i'%(i_target+1),
                                                     fontweight='bold',fontsize=15)
                if i_target == trgt.shape[1]-1:
                    ax[i_target,i_output].set_xlabel('Detector %i'%(i_output+1),
                                                     fontweight='bold',fontsize=15)
                ax[i_target,i_output].grid()
        
        return fig
    
    
    def analysis_nu_sweep(self, data, trgt, trn_info=None, min_nu=0.1, max_nu=0.9, nu_step=0.1, num_cores=1):
        print 'SVMNoveltyDetection analysis nu sweep function'
        
        if min_nu < 0.0 or min_nu >1.0:
            print 'Invalid min nu value...'
            return None

        if max_nu < 0.0 or max_nu >1.0:
            print 'Invalid max nu value...'
            return None

        if max_nu < min_nu:
            print 'Invalid max nu should be greater than max_nu...'
            return None


        # checar se a analise ja foi feita
        min_nu_str = ("%.5f"%(min_nu)).replace('.','_')
        max_nu_str = ("%.5f"%(max_nu)).replace('.','_')
        step_nu_str = ("%.5f"%(nu_step)).replace('.','_')
        file_name = '%s/%s_%s_analysis_nu_sweep_min_%s_max_%s_step_%s.jbl'%(self.anal_path,
                                                                            self.trn_info.date,
                                                                            self.name,
                                                                            min_nu_str,
                                                                            max_nu_str,
                                                                            step_nu_str)

        if not os.path.exists(file_name):
            nu_values = np.arange(min_nu,max_nu+nu_step/2.0,nu_step)
            
            qtd_folds = self.trn_info.n_folds
            qtd_classes = trgt.shape[1]
            qtd_nu = nu_values.shape[0]

            # to be easy to compare
            trgt_num = trgt.argmax(axis=1)
    
            # Figures of Merit
            # qtd_classes -1 = all known classes
            # qtd_classes = possible nolvety classes
            eff_known_class = np.zeros([qtd_folds,qtd_classes,qtd_classes-1,qtd_nu])
            tri_known_class = np.zeros([qtd_folds,qtd_classes,qtd_nu])
            eff_novelty = np.zeros([qtd_folds,qtd_classes,qtd_nu])
            
            for ifold in range(self.trn_info.n_folds):
                for i_novelty_class in range(trgt.shape[1]):
                    for i_nu_value in nu_values:
                        classifiers = self.train(data, trgt,
                                                 novelty_class=i_novelty_class,
                                                 trn_info=trn_info,
                                                 nu_value=i_nu_value,
                                                 fold=ifold)
                        for iclass in range(len(classifiers)):
                            if not iclass == i_novelty_class:
                                output = classifiers[iclass].predict(data)
                                eff_aux = float(sum(output[trgt_num==iclass]==1))/float(sum(trgt_num==iclass))
                                eff_known_class[ifold,i_novelty_class,iclass-(iclass>i_novelty_class),i_nu] = eff_aux
                            else:
                                # novelty detection
                                output = classifiers[i_novelty_class].predict(data)
                                eff_aux = float(sum(output[trgt_num==i_novelty_class]==-1))/float(sum(trgt_num==i_novelty_class))
                                eff_novelty[ifold,i_novelty_class,i_nu_value] = eff_aux
                                
                                # trigger
                                eff_aux = float(sum(output[trgt_num!=i_novelty_class]==1))/float(sum(trgt_num!=i_novelty_class))
                                tri_known_class[ifold,i_novelty_class,i_nu_value] = eff_aux
            joblib.dump([nu_values,eff_known_class,eff_novelty,tri_known_class],file_name,compress=9)
        else:
            [nu_values,eff_known_class,eff_novelty,tri_known_class] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(20,20),nrows=2, ncols=2)

        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['legend.handlelength'] = 3
        plt.rcParams['legend.borderpad'] = 0.3

        m_colors = ['b', 'r', 'g', 'y']
        m_fontsize = 18
        line_width = 3.5

        for novelty_class in range(trgt.shape[1]):
            axis = plt.subplot(2,2,novelty_class+1)
            plt.title('Classifier Eff. - Novelty: '+str(novelty_class+1), fontsize= m_fontsize, fontweight="bold")
            if novelty_class > -1:
                plt.xlabel(r'$\nu$ values', fontsize= m_fontsize, fontweight="bold")
            plt.ylabel('Efficiency (%)', fontsize= m_fontsize, fontweight="bold")
            m_leg = []

            for known_class in range(trgt.shape[1]):
                if known_class == novelty_class:
                    continue
                #print "Novelty Class %i - Known Class %i - Index %i"%(novelty_class,known_class,known_class-(known_class>novelty_class))
                plot_data=eff_known_class[:,novelty_class,known_class-(known_class>novelty_class),:]
                axis.errorbar(nu_values,
                              100*np.mean(plot_data,axis=0),
                              100*np.std(plot_data,axis=0),marker='o',
                              color=m_colors[known_class],alpha=0.5,
                              linewidth=line_width)
                m_leg.append('Class %i Eff.'%(known_class))

            plot_data=eff_novelty[:,novelty_class,:]
            axis.errorbar(nu_values,
                          100*np.mean(plot_data,axis=0),
                          100*np.std(plot_data,axis=0),marker='o',
                          color='k',alpha=0.5,
                          linewidth=line_width)
            m_leg.append('Novelty Eff.')

            plot_data=tri_known_class[:,novelty_class,:]
            axis.errorbar(nu_values,
                          100*np.mean(plot_data,axis=0),
                          100*np.std(plot_data,axis=0),marker='o',
                          color='k', ls=':',alpha=0.5,
                          linewidth=line_width)
            m_leg.append('Trigger')


            # graphical assusts
            axis.set_ylim([0.0, 115])
            axis.set_yticks([x for x in range(0,101,5)])
        
            axis.set_xlim([np.min(nu_values), np.max(nu_values)])
            axis.set_xticks(nu_values)
            axis.set_xticklabels(nu_values,rotation=45)

            axis.grid()
            axis.legend(m_leg, loc='upper right',ncol=3)
    
    
        return fig




class PCASVMNoveltyDetection(SVMNoveltyDetection):
    def preprocess(self, data, trgt, novelty_class=0, trn_info=None, fold=0):
        print 'SVMNoveltyDetection preprocess function'


class CnnNoveltyAnalysis(object):
    def __init__(self, ncv_obj, trn_params_mapping, package_name, analysis_name, class_labels, novelty_classes = None):
        if novelty_classes is None:
            novelty_classes = class_labels

        self.an_path = package_name + '/' + analysis_name
        self.resultspath = package_name
        self.ncv_obj = ncv_obj
        self.class_labels = class_labels
        self.ModelsAnalysis = dict()
        for cls in novelty_classes.values():
            self.ModelsAnalysis[cls] = CnnClassificationAnalysis(ncv_obj,
                                                                 trn_params_mapping,
                                                                 package_name + '/%s' % cls,
                                                                 '../%s' % cls + analysis_name,
                                                                 class_labels)

        self.nv_predictions = None

    def _recover(self, trn_params_mapping, data, trgt):
        import keras.backend as K
        for nv_i, nv_cls in self.class_labels.items():
            for cv_name, cv in self.ncv_obj.cv.items():
                fold_info = SonarRunsKFold(10, dev=False,
                                           split_All={'ClassA': 4, 'ClassB': 4, 'ClassC': 4, 'ClassD': 4},
                                           validation_runs={'ClassA': 2, 'ClassB': 1, 'ClassC': 1, 'ClassD': 1},
                                           val_share=3)
                for name, trnparams in trn_params_mapping.items():
                    print name
                    path = self.resultspath + '/%s/' % nv_cls + trnparams.getParamPath() + '/' + cv_name
                    model = KerasModel(trnparams)
                    prediction = np.empty(10, dtype=np.ndarray)
                    for i_fold in os.listdir(path + '/best_states'):
                        fold_file = path + '/best_states/' + i_fold
                        model.load(fold_file)
                        for i_c, (train,test) in enumerate(cv):
                            if int(i_fold[0]) == i_c:
                                x_test, y_test  = lofar2image(all_data=data,
                                                                all_trgt=trgt,
                                                                index_info=test,
                                                                window_size=int(name),
                                                                stride=int(name),
                                                                run_indices_info=fold_info)


                                pred = model.predict(x_test)
                                pred = np.concatenate([pred[:, :nv_i],
                                                     np.repeat(np.nan, pred.shape[0])[:, np.newaxis],
                                                     pred[:, nv_i:],
                                                     np.array(y_test[:,np.newaxis], dtype=np.int)],
                                                    axis=1)
                                prediction[int(i_fold[0])] = pred

                    column_names = self.class_labels.values()
                    column_names.append('Label')
                    predictions_pd = [pd.DataFrame(fold_prediction, columns=column_names,
                                                    index=pd.MultiIndex.from_product(
                                                        [['fold_%i' % i_fold], [nv_i],
                                                         range(fold_prediction.shape[0])]))
                                       for i_fold, fold_prediction in enumerate(prediction)]

                    pd_pred = pd.concat(predictions_pd, axis=0)
                    pd_pred.to_csv(path + '/predictions.csv')

                    if K.backend() == 'tensorflow':  # solve tf memory leak
                        K.clear_session()

                        #model.load(self.resultspath + '/%s' % nv_cls + trnparams.getParamPath())


    def stdAnalysis(self, data, trgt):
        for modelAnalysis in self.ModelsAnalysis.values():
            modelAnalysis.stdAnalysis(data, trgt)

    def getNvPrediction(self):
        nv_predictions = pd.DataFrame()
        inverse_labels = {value: key for key,value in self.class_labels.items()}

        for nv_cls, modelAnalysis in self.ModelsAnalysis.items():
            for model_name, model in modelAnalysis.modelsData.items():
                for cv_name, predictions in model.predictions.items():
                    predictions['Model'] = model_name
                    predictions['CV'] = cv_name
                    predictions['Novelty'] = inverse_labels[nv_cls]
                    nv_predictions = nv_predictions.append(predictions)

        self.nv_predictions = nv_predictions
        # for model_name in nv_scores.keys():
        #     for cv_name in nv_scores[model_name].keys():
        #         nv_scores[model_name][cv_name] = pd.concat(nv_scores[model_name][cv_name], axis=1)
        #
        #         # print nv_scores[model_name][cv_name]


    def _get_class_int(self, x, v):
        x_out = x.filter(regex=('Class*'))
        max_column = np.nanargmax(x_out.values)
        max_value = x_out.max(skipna=True)

        return max_column if max_value > v else np.argwhere(pd.isnull(x_out).values == True)[0][0]

    def _getThresGrid(self, v_resolution):
        threshold_values = np.linspace(.0, 1., v_resolution, dtype=np.float16)
        threshold_values_zoom = np.linspace(.95, 1., v_resolution, dtype=np.float16)
        if not exists(self.an_path + '/%i_threshold_pd.csv' % v_resolution):
            threshold_pd = pd.DataFrame(np.array([self.nv_predictions.apply(lambda x: self._get_class_int(x, t), axis=1)
                                                  for t in threshold_values]).T,
                                        columns=threshold_values,
                                        index=self.nv_predictions.index)

            threshold_pd['Label'] = self.nv_predictions['Label'].values
            threshold_pd['CV'] = self.nv_predictions['CV'].values
            threshold_pd['Novelty'] = self.nv_predictions['Novelty'].values
            threshold_pd['Model'] = self.nv_predictions['Model'].values

            threshold_pd_zoom = pd.DataFrame(np.array([self.nv_predictions.apply(lambda x: self._get_class_int(x, t), axis=1)
                                                       for t in threshold_values_zoom]).T,
                                             columns=threshold_values_zoom,
                                             index=self.nv_predictions.index)
            threshold_pd_zoom['Label'] = self.nv_predictions['Label'].values
            threshold_pd_zoom['CV'] = self.nv_predictions['CV'].values
            threshold_pd_zoom['Novelty'] = self.nv_predictions['Novelty'].values
            threshold_pd_zoom['Model'] = self.nv_predictions['Model'].values

            threshold_pd.to_csv(self.an_path + '/%i_threshold_pd.csv' % v_resolution)
            threshold_pd_zoom.to_csv(self.an_path + '/%i_threshold_pd_zoom.csv' % v_resolution)
        else:
            print ('File found')

            threshold_pd = pd.read_csv(self.an_path + '/%i_threshold_pd.csv' % v_resolution, index_col=[0,1])
            threshold_pd_zoom = pd.read_csv(self.an_path + '/%i_threshold_pd_zoom.csv' % v_resolution, index_col=[0, 1])
            # inverse_labels = {value: key for key, value in self.class_labels.items()}
            # threshold_pd['Novelty'] = np.array([inverse_labels[cls] for cls in threshold_pd['Novelty'].values])
            # threshold_pd_zoom['Novelty'] = np.array([inverse_labels[cls] for cls in threshold_pd_zoom['Novelty'].values])
            # threshold_pd.to_csv(self.an_path + '/%i_threshold_pd.csv' % v_resolution)
            # threshold_pd_zoom.to_csv(self.an_path + '/%i_threshold_pd_zoom.csv' % v_resolution)

        return threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom

    def _iterNvRate(self, v_res):
        threshold_pd, threshold_pd_zoom,threshold_values, threshold_values_zoom = self._getThresGrid(v_res)
        nv_scores = pd.DataFrame()
        nv_scores_zoom = pd.DataFrame()

        for (model_name, model_thres), (_, model_thres_zoom) in zip(threshold_pd.groupby(by='Model'),
                                                                    threshold_pd_zoom.groupby(by='Model')):
            for (cv_name, cv_thres), (_, cv_thres_zoom) in zip(model_thres.groupby(by='CV'),
                                                               model_thres_zoom.groupby(by='CV')):
                for (novelty_cls, folds_thresh_pd), (_, folds_thresh_pd_zoom) \
                        in zip(cv_thres.groupby(by='Novelty'),
                               cv_thres_zoom.groupby(by='Novelty')):

                    known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]

                    nv_rate_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                              .apply(lambda x: recall_score(group['Label'].values,
                                                                            x.values)[novelty_cls], axis=0)
                                               for name, group in folds_thresh_pd.groupby(level='Fold')])

                    nv_rate_matrix_zoom = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                                   .apply(lambda x: recall_score(group['Label']
                                                                                 .values, x.values)[novelty_cls], axis=0)
                                                    for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])

                    nv_rate_stats = pd.DataFrame({'mean': nv_rate_matrix.mean(axis=0),
                                                  'std': nv_rate_matrix.std(axis=0)})
                    nv_rate_stats_zoom = pd.DataFrame({'mean': nv_rate_matrix_zoom.mean(axis=0),
                                                       'std': nv_rate_matrix_zoom.std(axis=0)})

                    nv_rate_stats['Model'] = model_name
                    nv_rate_stats['CV'] = cv_name
                    nv_rate_stats['Novelty'] = novelty_cls

                    nv_rate_stats_zoom['Model'] = model_name
                    nv_rate_stats_zoom['CV'] = cv_name
                    nv_rate_stats_zoom['Novelty'] = novelty_cls

                    nv_scores = nv_scores.append(nv_rate_stats)
                    nv_scores_zoom = nv_scores_zoom.append(nv_rate_stats_zoom)
        return nv_scores, nv_scores_zoom


    def _iterTriggerRate(self, v_res):
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger= pd.DataFrame()
        trigger_zoom = pd.DataFrame()

        for (model_name, model_thres), (_, model_thres_zoom) in zip(threshold_pd.groupby(by='Model'),
                                                                    threshold_pd_zoom.groupby(by='Model')):
            for (cv_name, cv_thres), (_, cv_thres_zoom) in zip(model_thres.groupby(by='CV'),
                                                               model_thres_zoom.groupby(by='CV')):
                for (novelty_cls, folds_thresh_pd), (_, folds_thresh_pd_zoom) \
                        in zip(cv_thres.groupby(by='Novelty'),
                               cv_thres_zoom.groupby(by='Novelty')):

                    known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]

                    trigger_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                              .apply(lambda x: trigger_score(group['Label'].values,
                                                                             x.values, novelty_cls),
                                                     axis=0)
                                               for name, group in folds_thresh_pd.groupby(level='Fold')])

                    trigger_matrix_zoom = np.array(
                        [group.drop(columns=['Novelty', 'CV', 'Model', 'Label']).apply(lambda x: trigger_score(group['Label'].values,
                                                                                                      x.values, novelty_cls), axis=0)
                         for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])


                    trigger_stats = pd.DataFrame({'mean': trigger_matrix.mean(axis=0),
                                             'std': trigger_matrix.std(axis=0)})
                    trigger_stats_zoom = pd.DataFrame({'mean': trigger_matrix_zoom.mean(axis=0),
                                                  'std': trigger_matrix_zoom.std(axis=0)})

                    trigger_stats['Model'] = model_name
                    trigger_stats['CV'] = cv_name
                    trigger_stats['Novelty'] = novelty_cls
                    trigger_stats_zoom['Model'] = model_name
                    trigger_stats_zoom['CV'] = cv_name
                    trigger_stats_zoom['Novelty'] = novelty_cls

                    trigger = trigger.append(trigger_stats)
                    trigger_zoom = trigger_zoom.append(trigger_stats_zoom)
        return trigger, trigger_zoom

    def _iterSpRate(self, v_res):
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger= pd.DataFrame()
        trigger_zoom = pd.DataFrame()

        for (model_name, model_thres), (_, model_thres_zoom) in zip(threshold_pd.groupby(by='Model'),
                                                                    threshold_pd_zoom.groupby(by='Model')):
            for (cv_name, cv_thres), (_, cv_thres_zoom) in zip(model_thres.groupby(by='CV'),
                                                               model_thres_zoom.groupby(by='CV')):
                for (novelty_cls, folds_thresh_pd), (_, folds_thresh_pd_zoom) \
                        in zip(cv_thres.groupby(by='Novelty'),
                               cv_thres_zoom.groupby(by='Novelty')):

                    known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]

                    recall_tensor = [np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                              .apply(lambda x: recall_score(group['Label']
                                                                            .values, x.values)[cls], axis=0)
                                               for name, group in folds_thresh_pd.groupby(level='Fold')])
                                     for cls in known_cls]

                    recall_tensor_zoom = [np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                                   .apply(lambda x: recall_score(group['Label']
                                                                                 .values, x.values)[cls], axis=0)
                                                    for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])
                                          for cls in known_cls]

                    trigger_matrix = np.apply_along_axis(lambda x: spIndex(x, 3), 0, np.array(recall_tensor))
                    trigger_matrix_zoom = np.apply_along_axis(lambda x: spIndex(x, 3), 0, np.array(recall_tensor_zoom))


                    trigger_stats = pd.DataFrame({'mean': trigger_matrix.mean(axis=0),
                                             'std': trigger_matrix.std(axis=0)})
                    trigger_stats_zoom = pd.DataFrame({'mean': trigger_matrix_zoom.mean(axis=0),
                                                  'std': trigger_matrix_zoom.std(axis=0)})

                    trigger_stats['Model'] = model_name
                    trigger_stats['CV'] = cv_name
                    trigger_stats['Novelty'] = novelty_cls
                    trigger_stats_zoom['Model'] = model_name
                    trigger_stats_zoom['CV'] = cv_name
                    trigger_stats_zoom['Novelty'] = novelty_cls

                    trigger = trigger.append(trigger_stats)
                    trigger_zoom = trigger_zoom.append(trigger_stats_zoom)
        return trigger, trigger_zoom

    def plotTriggerNvRate(self, v_res):
        colors = {0: 'b', 1: 'g', 2: 'y', 3: 'r'}
        linst = {0: '-', 1: ':', 2: '-.', 3: '--'}
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger, trigger_zoom = self._iterTriggerRate(v_res)
        nv_rate, nv_rate_zoom = self._iterNvRate(v_res)
        def mask(df, key, value):
            return df.loc[df[key] == value]

        pd.DataFrame.mask = mask


        for (novelty_cls, nv_trigger) in trigger.groupby(by='Novelty'):
            for cv_name, cv_trigger in nv_trigger.groupby(by='CV'):
                roc_fig = plt.figure(figsize=(6, 6))
                roc_ax = plt.gca()
                for i, (model_name, model_trigger) in enumerate(cv_trigger.groupby(by='Model')):
                    # model_nv_rate = nv_rate.mask('CV', cv_name) \
                    #     .mask('Novelty', novelty_cls)\
                    #     .mask('Model', model_name)
                    #
                    # model_trigger_zoom = trigger_zoom.mask('CV', cv_name) \
                    #     .mask('Novelty', novelty_cls)\
                    #     .mask('Model', model_name)
                    #
                    # model_nv_rate_zoom = nv_rate_zoom.mask('CV', cv_name) \
                    #     .mask('Novelty', novelty_cls)\
                    #     .mask('Model', model_name)
                    #
                    # x = np.concatenate([threshold_values[threshold_values < 0.95],
                    #                         threshold_values_zoom])
                    # trigger_y = np.hstack([model_trigger[:, threshold_values < 0.95],
                    #                        model_trigger_zoom[:, :]])
                    # nv_y = np.hstack([model_nv_rate[:, threshold_values < 0.95],
                    #                       model_nv_rate_zoom[:, :]])

                    roc_ax.set_title('Trigger x Novelty rate', fontsize=18)

                    roc_ax.plot(model_nv_rate['mean'].values, model_trigger['mean'].values, color=colors.values()[i],
                                linestyle=linst.values()[i])
                    roc_ax.fill_between(model_nv_rate['mean'].values, model_trigger['mean'] + model_trigger['std'],
                                        model_trigger['mean'] - model_trigger['std'],
                                        alpha=0.15, color=colors.values()[i], label=model_name)
                    roc_ax.set_ylabel('Trigger', fontsize=15)
                    roc_ax.set_xlabel('Novelty Rate', fontsize=15)

                lines, labels = roc_ax.get_legend_handles_labels()
                roc_ax.legend(lines, labels, loc='lower left', ncol=2)

                savepath = self.an_path + '/%s/' % cv_name
                if not exists(savepath):
                    mkdir(savepath)
                roc_fig.savefig(savepath + 'nv_t_trigger_%i.pdf' % novelty_cls)


    def getAUC(self, v_res):
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger = pd.DataFrame()
        trigger_zoom = pd.DataFrame()
        trigger_auc = pd.DataFrame(columns=['mean', 'std'])
        nv_pd_auc = pd.DataFrame(columns=['mean', 'std'])

        for (model_name, model_thres), (_, model_thres_zoom) in zip(threshold_pd.groupby(by='Model'),
                                                                    threshold_pd_zoom.groupby(by='Model')):
            for (cv_name, cv_thres), (_, cv_thres_zoom) in zip(model_thres.groupby(by='CV'),
                                                               model_thres_zoom.groupby(by='CV')):
                for (novelty_cls, folds_thresh_pd), (_, folds_thresh_pd_zoom) \
                        in zip(cv_thres.groupby(by='Novelty'),
                               cv_thres_zoom.groupby(by='Novelty')):
                    known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]

                    trigger_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                              .apply(lambda x: trigger_score(group['Label'].values,
                                                                             x.values, novelty_cls),
                                                     axis=0)
                                               for name, group in folds_thresh_pd.groupby(level='Fold')])

                    trigger_matrix_zoom = np.array(
                        [group.drop(columns=['Novelty', 'CV', 'Model', 'Label']).apply(
                            lambda x: trigger_score(group['Label'].values,
                                                    x.values, novelty_cls), axis=0)
                         for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])

                    known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]

                    nv_rate_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                              .apply(lambda x: recall_score(group['Label'].values,
                                                                            x.values)[novelty_cls], axis=0)
                                               for name, group in folds_thresh_pd.groupby(level='Fold')])

                    nv_rate_matrix_zoom = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
                                                   .apply(lambda x: recall_score(group['Label']
                                                                                 .values, x.values)[novelty_cls],
                                                          axis=0)
                                                    for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])

                    nv_rate_stats = pd.DataFrame({'mean': nv_rate_matrix.mean(axis=0),
                                                  'std': nv_rate_matrix.std(axis=0)})
                    nv_rate_stats_zoom = pd.DataFrame({'mean': nv_rate_matrix_zoom.mean(axis=0),
                                                       'std': nv_rate_matrix_zoom.std(axis=0)})

                    auc_x = np.concatenate([threshold_values[threshold_values < 0.95],
                                            threshold_values_zoom])
                    auc_sp_y = np.hstack([trigger_matrix[:, threshold_values < 0.95],
                                          trigger_matrix_zoom[:, :]])
                    auc_nv_y = np.hstack([nv_rate_matrix[:, threshold_values < 0.95],
                                          nv_rate_matrix_zoom[:, :]])

                    sp_auc = np.apply_along_axis(lambda x: auc(auc_x, x), axis=1, arr=auc_sp_y)
                    nv_auc = np.apply_along_axis(lambda x: auc(auc_x, x), axis=1, arr=auc_nv_y)

                    # print model_name
                    # print novelty_cls
                    # # print sp_auc
                    # # print nv_auc
                    # print '%f+-%f' % (sp_auc.mean(), sp_auc.std())
                    # print '%f+-%f' % (nv_auc.mean(), nv_auc.std())
                    t_auc = pd.DataFrame({'mean':sp_auc.mean(), 'std':sp_auc.std()},
                                         index=pd.MultiIndex.from_tuples([(model_name, novelty_cls)]))
                    trigger_auc.append(t_auc)
                    pd_auc = pd.DataFrame({'mean': nv_auc.mean(), 'std': nv_auc.std()},
                                         index=pd.MultiIndex.from_tuples([(model_name, novelty_cls)]))
                    trigger_auc = trigger_auc.append(t_auc)
                    nv_pd_auc = nv_pd_auc.append(pd_auc)

        # fig = plt.figure(figsize=(6, 6))
        # auc_ax = plt.gca()
        #
        # auc_ax.plot(, trigger_auc['mean'].values, color='black')
        # auc_ax.errorbar(threshold_values, cv_trigger['mean'], cv_trigger['std'],
        #                 color='black', label='Trigger')

        print trigger_auc

    def getSpOp(self, v_res):
        trigger, trigger_zoom = self._iterSpRate(v_res)
        nv_rate, nv_rate_zoom = self._iterNvRate(v_res)

        def mask(df, key, value):
            return df.loc[df[key] == value]

        pd.DataFrame.mask = mask
        for (novelty_cls, nv_trigger) in trigger.groupby(by='Novelty'):
            for (model_name, model_trigger) in nv_trigger.groupby(by='Model'):
                for cv_name, trigger_stats in model_trigger.groupby(by='CV'):

                    trigger_stats_zoom = trigger_zoom.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)

                    nv_rate_stats = nv_rate.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)

                    nv_rate_stats_zoom = nv_rate_zoom.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)
                    print model_name
                    print  novelty_cls
                    for v in [0.0, 0.5]:
                        self.trigger_zero = trigger_stats['mean'][nv_rate_stats['mean'] <= v].values
                        self.nv_rate_stats = nv_rate_stats
                        lin_mean = (trigger_stats['mean'][nv_rate_stats['mean'].values > v].values[0] +
                                    trigger_stats['mean'][nv_rate_stats['mean'].values <= v].values[-1]) / 2
                        lin_std = (trigger_stats['std'][nv_rate_stats['mean'].values > v].values[0] +
                                   trigger_stats['std'][nv_rate_stats['mean'].values <= v].values[-1]) / 2
                        # print trigger_stats
                        print trigger_stats['mean'][nv_rate_stats['mean'].values <= v]
                        print trigger_stats['std'][nv_rate_stats['mean'].values <= v]
                        print '\t%.2f:\t ' % v + \
                              '%f +- %f' % (lin_mean, lin_std)
                    for v in [0.75]:
                        lin_mean = (trigger_stats_zoom['mean'][nv_rate_stats_zoom['mean'].values > v].values[0] +
                                    trigger_stats_zoom['mean'][nv_rate_stats_zoom['mean'].values <= v].values[-1]) / 2
                        lin_std = (trigger_stats_zoom['std'][nv_rate_stats_zoom['mean'].values > v].values[0] +
                                   trigger_stats_zoom['std'][nv_rate_stats_zoom['mean'].values <= v].values[-1]) / 2
                        print '\t%.2f:\t ' % v + \
                              '%f +- %f' % (lin_mean, lin_std)

    def getTriggerOp(self, v_res):
        trigger, trigger_zoom = self._iterTriggerRate(v_res)
        nv_rate, nv_rate_zoom = self._iterNvRate(v_res)

        def mask(df, key, value):
            return df.loc[df[key] == value]

        pd.DataFrame.mask = mask
        for (novelty_cls, nv_trigger) in trigger.groupby(by='Novelty'):
            for (model_name, model_trigger) in nv_trigger.groupby(by='Model'):
                for cv_name, trigger_stats in model_trigger.groupby(by='CV'):

                    trigger_stats_zoom = trigger_zoom.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)

                    nv_rate_stats = nv_rate.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)

                    nv_rate_stats_zoom = nv_rate_zoom.mask('Model', model_name) \
                        .mask('CV', cv_name) \
                        .mask('Novelty', novelty_cls)
                    print model_name
                    print  novelty_cls
                    for v in [0.0, 0.5]:
                        self.trigger_zero = trigger_stats['mean'][nv_rate_stats['mean'] <= v].values
                        self.nv_rate_stats = nv_rate_stats
                        lin_mean = (trigger_stats['mean'][nv_rate_stats['mean'].values > v].values[0] +
                                    trigger_stats['mean'][nv_rate_stats['mean'].values <= v].values[-1]) / 2
                        lin_std = (trigger_stats['std'][nv_rate_stats['mean'].values > v].values[0] +
                                   trigger_stats['std'][nv_rate_stats['mean'].values <= v].values[-1]) / 2
                        # print trigger_stats
                        print '\t%.2f:\t ' % v + \
                              '%f +- %f' % (lin_mean, lin_std)
                    for v in [0.75]:
                        lin_mean = (trigger_stats_zoom['mean'][nv_rate_stats_zoom['mean'].values > v].values[0] +
                                    trigger_stats_zoom['mean'][nv_rate_stats_zoom['mean'].values <= v].values[-1]) / 2
                        lin_std = (trigger_stats_zoom['std'][nv_rate_stats_zoom['mean'].values > v].values[0] +
                                   trigger_stats_zoom['std'][nv_rate_stats_zoom['mean'].values <= v].values[-1]) / 2
                        print '\t%.2f:\t ' % v + \
                              '%f +- %f' % (lin_mean, lin_std)


    def plotTriggerThresholds(self, v_res):
        colors = {0: 'b', 1: 'g', 2: 'y', 3: 'r'}
        linst = {0: '-', 1: ':', 2: '-.', 3: '--'}
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger, trigger_zoom = self._iterTriggerRate(v_res)
        nv_rate, nv_rate_zoom = self._iterNvRate(v_res)
        def mask(df, key, value):
            return df.loc[df[key] == value]

        pd.DataFrame.mask = mask
        for (novelty_cls, nv_trigger) in trigger.groupby(by='Novelty'):
            for (model_name, model_trigger) in nv_trigger.groupby(by='Model'):
                for cv_name, cv_trigger in model_trigger.groupby(by='CV'):

                    cv_trigger_zoom = trigger_zoom.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)

                    cv_nv_rate = nv_rate.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)


                    cv_nv_rate_zoom = nv_rate_zoom.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)

                    fig = plt.figure(figsize=(6, 6))
                    nv_ax = plt.gca()
                    sp_ax = nv_ax.twinx()

                    sp_ax.plot(threshold_values, cv_trigger['mean'].values, color='black')
                    sp_ax.fill_between(threshold_values, cv_trigger['mean'] + cv_trigger['std'],
                                       cv_trigger['mean'] - cv_trigger ['std'], color='black',
                                       alpha=0.3, label='Trigger')

                    nv_ax.plot(threshold_values, cv_nv_rate['mean'].values, color='indigo', linestyle=':')
                    nv_ax.fill_between(threshold_values, cv_nv_rate['mean'] + cv_nv_rate['std'],
                                       cv_nv_rate['mean'] - cv_nv_rate['std'],
                                       alpha=0.3, color='indigo', label='Novelty rate')
                    nv_ax.set_ylabel('Novelty rate', fontsize=15)
                    sp_ax.set_ylabel('SP index', fontsize=15)
                    nv_ax.set_xlabel('Threshold', fontsize=15)

                    for ax in [nv_ax, sp_ax]:
                        ax.set_ylim(0, 1.12)

                    plt.title('Novelty detection for %s' % self.class_labels[novelty_cls], fontsize=18)
                    lines, labels = nv_ax.get_legend_handles_labels()
                    lines2, labels2 = sp_ax.get_legend_handles_labels()
                    nv_ax.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2)

                    ax2 = plt.axes([0.22, 0.33, 0.50, 0.35])

                    ax2.plot(threshold_values_zoom, cv_trigger_zoom['mean'].values, color='black')
                    ax2.fill_between(threshold_values_zoom,
                                     cv_trigger_zoom['mean'] + cv_trigger_zoom['std'],
                                     cv_trigger_zoom['mean'] - cv_trigger_zoom['std'],
                                     color='black',
                                     alpha=0.3)

                    ax2.plot(threshold_values_zoom, cv_nv_rate_zoom['mean'].values, color='indigo', linestyle=':')
                    ax2.fill_between(threshold_values_zoom, cv_nv_rate_zoom['mean'] + cv_nv_rate_zoom['std'],
                                     cv_nv_rate_zoom['mean'] - cv_nv_rate_zoom['std'],
                                     color='indigo', alpha=0.3)
                    ax2.set_xlim(0.95, 1)
                    ax2.set_ylim(.2, 1)

                    savepath = self.an_path + '/%s/%s/' % (model_name, cv_name)
                    if not exists(savepath):
                        mkdir(savepath)
                    plt.savefig(savepath + 'nv_t_trigger_%i.pdf' % novelty_cls,
                                bbox_inches='tight')
                    plt.close(fig)

    def plotSPThresholds(self, v_res):
        colors = {0: 'b', 1: 'g', 2: 'y', 3: 'r'}
        linst = {0: '-', 1: ':', 2: '-.', 3: '--'}
        threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)

        trigger, trigger_zoom = self._iterSpRate(v_res)
        nv_rate, nv_rate_zoom = self._iterNvRate(v_res)
        def mask(df, key, value):
            return df.loc[df[key] == value]

        pd.DataFrame.mask = mask
        for (novelty_cls, nv_trigger) in trigger.groupby(by='Novelty'):
            for (model_name, model_trigger) in nv_trigger.groupby(by='Model'):
                for cv_name, cv_trigger in model_trigger.groupby(by='CV'):

                    cv_trigger_zoom = trigger_zoom.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)

                    cv_nv_rate = nv_rate.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)


                    cv_nv_rate_zoom = nv_rate_zoom.mask('Model', model_name)\
                        .mask('CV', cv_name)\
                        .mask('Novelty', novelty_cls)

                    fig = plt.figure(figsize=(6, 6))
                    nv_ax = plt.gca()
                    sp_ax = nv_ax.twinx()

                    sp_ax.plot(threshold_values, cv_trigger['mean'].values, color='black')
                    sp_ax.fill_between(threshold_values, cv_trigger['mean'] + cv_trigger['std'],
                                       cv_trigger['mean'] - cv_trigger ['std'], color='black',
                                       alpha=0.3, label='SP index')

                    nv_ax.plot(threshold_values, cv_nv_rate['mean'].values, color='crimson', linestyle=':')
                    nv_ax.fill_between(threshold_values, cv_nv_rate['mean'] + cv_nv_rate['std'],
                                       cv_nv_rate['mean'] - cv_nv_rate['std'],
                                       alpha=0.3, color='crimson', label='Novelty rate')
                    nv_ax.set_ylabel('Novelty rate', fontsize=15)
                    sp_ax.set_ylabel('SP index', fontsize=15)
                    nv_ax.set_xlabel('Threshold', fontsize=15)

                    for ax in [nv_ax, sp_ax]:
                        ax.set_ylim(0, 1.12)

                    plt.title('Novelty detection for %s' % self.class_labels[novelty_cls], fontsize=18)
                    lines, labels = nv_ax.get_legend_handles_labels()
                    lines2, labels2 = sp_ax.get_legend_handles_labels()
                    nv_ax.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2)

                    ax2 = plt.axes([0.22, 0.33, 0.50, 0.35])

                    ax2.plot(threshold_values_zoom, cv_trigger_zoom['mean'].values, color='black')
                    ax2.fill_between(threshold_values_zoom,
                                     cv_trigger_zoom['mean'] + cv_trigger_zoom['std'],
                                     cv_trigger_zoom['mean'] - cv_trigger_zoom['std'],
                                     color='black',
                                     alpha=0.3)

                    ax2.plot(threshold_values_zoom, cv_nv_rate_zoom['mean'].values, color='crimson', linestyle=':')
                    ax2.fill_between(threshold_values_zoom, cv_nv_rate_zoom['mean'] + cv_nv_rate_zoom['std'],
                                     cv_nv_rate_zoom['mean'] - cv_nv_rate_zoom['std'],
                                     color='crimson', alpha=0.3)
                    ax2.set_xlim(0.95, 1)
                    ax2.set_ylim(.2, 1)

                    savepath = self.an_path + '/%s/%s/' % (model_name, cv_name)
                    if not exists(savepath):
                        mkdir(savepath)
                    plt.savefig(savepath + 'nv_t_sp_%i.pdf' % novelty_cls,
                                bbox_inches='tight')
                    plt.savefig(savepath + 'nv_t_sp_%i.png' % novelty_cls,
                                bbox_inches='tight')
                    plt.close(fig)


    # def plotTriggerThresholds(self, v_res):
    #     colors = {0: 'b', 1: 'g', 2: 'y', 3: 'r'}
    #     linst = {0: '-', 1: ':', 2: '-.', 3: '--'}
    #     threshold_pd, threshold_pd_zoom, threshold_values, threshold_values_zoom = self._getThresGrid(v_res)
    #
    #     for (model_name, model_thres), (_, model_thres_zoom) in zip(threshold_pd.groupby(by='Model'),
    #                                                                 threshold_pd_zoom.groupby(by='Model')):
    #         for (cv_name, cv_thres), (_, cv_thres_zoom) in zip(model_thres.groupby(by='CV'),
    #                                                            model_thres_zoom.groupby(by='CV')):
    #             for (novelty_cls, folds_thresh_pd), (_, folds_thresh_pd_zoom) \
    #                     in zip(cv_thres.groupby(by='Novelty'),
    #                            cv_thres_zoom.groupby(by='Novelty')):
    #
    #                 known_cls = [value for value in self.class_labels.keys() if value != novelty_cls]
    #
    #                 trigger_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
    #                                           .apply(lambda x: trigger_score(group['Label'].values,
    #                                                                          x.values, novelty_cls),
    #                                                  axis=0)
    #                                            for name, group in folds_thresh_pd.groupby(level='Fold')])
    #
    #                 trigger_matrix_zoom = np.array(
    #                     [group.drop(columns=['Novelty', 'CV', 'Model', 'Label']).apply(lambda x: trigger_score(group['Label'].values,
    #                                                                                                   x.values, novelty_cls), axis=0)
    #                      for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])
    #
    #
    #                 nv_rate_matrix = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
    #                                           .apply(lambda x: recall_score(group['Label'].values,
    #                                                                         x.values)[novelty_cls], axis=0)
    #                                            for name, group in folds_thresh_pd.groupby(level='Fold')])
    #
    #                 nv_rate_matrix_zoom = np.array([group.drop(columns=['Novelty', 'CV', 'Model', 'Label'])
    #                                                .apply(lambda x: recall_score(group['Label']
    #                                                                              .values, x.values)[novelty_cls], axis=0)
    #                                                 for name, group in folds_thresh_pd_zoom.groupby(level='Fold')])
    #
    #                 nv_rate_stats = pd.DataFrame({'mean': nv_rate_matrix.mean(axis=0),
    #                                               'std': nv_rate_matrix.std(axis=0)})
    #                 nv_rate_stats_zoom = pd.DataFrame({'mean': nv_rate_matrix_zoom.mean(axis=0),
    #                                                    'std': nv_rate_matrix_zoom.std(axis=0)})
    #
    #                 sp_stats = pd.DataFrame({'mean': trigger_matrix.mean(axis=0),
    #                                          'std': trigger_matrix.std(axis=0)})
    #                 sp_stats_zoom = pd.DataFrame({'mean': trigger_matrix_zoom.mean(axis=0),
    #                                               'std': trigger_matrix_zoom.std(axis=0)})
    #
    #                 fig = plt.figure(figsize=(6, 6))
    #                 nv_ax = plt.gca()
    #                 sp_ax = nv_ax.twinx()
    #
    #                 sp_ax.plot(threshold_values, sp_stats['mean'].values, color='black')
    #                 sp_ax.fill_between(threshold_values, sp_stats['mean'] + sp_stats['std'],
    #                                    sp_stats['mean'] - sp_stats['std'], color='black',
    #                                    alpha=0.3, label='Trigger')
    #
    #                 nv_ax.plot(threshold_values, nv_rate_stats['mean'].values, color='indigo', linestyle=':')
    #                 nv_ax.fill_between(threshold_values, nv_rate_stats['mean'] + nv_rate_stats['std'],
    #                                    nv_rate_stats['mean'] - nv_rate_stats['std'],
    #                                    alpha=0.3, color='indigo', label='Novelty rate')
    #                 nv_ax.set_ylabel('Novelty rate', fontsize=15)
    #                 sp_ax.set_ylabel('SP index', fontsize=15)
    #                 nv_ax.set_xlabel('Threshold', fontsize=15)
    #
    #                 for ax in [nv_ax, sp_ax]:
    #                     ax.set_ylim(0, 1.08)
    #
    #                 plt.title('Novelty detection for %s' % self.class_labels[novelty_cls], fontsize=18)
    #                 lines, labels = nv_ax.get_legend_handles_labels()
    #                 lines2, labels2 = sp_ax.get_legend_handles_labels()
    #                 nv_ax.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2)
    #
    #                 ax2 = plt.axes([0.22, 0.33, 0.50, 0.35])
    #
    #                 ax2.plot(threshold_values_zoom, sp_stats_zoom['mean'].values, color='black')
    #                 ax2.fill_between(threshold_values_zoom,
    #                                  sp_stats_zoom['mean'] + sp_stats_zoom['std'],
    #                                  sp_stats_zoom['mean'] - sp_stats_zoom['std'],
    #                                  color='black',
    #                                  alpha=0.3)
    #
    #                 ax2.plot(threshold_values_zoom, nv_rate_stats_zoom['mean'].values, color='indigo', linestyle=':')
    #                 ax2.fill_between(threshold_values_zoom, nv_rate_stats_zoom['mean'] + nv_rate_stats_zoom['std'],
    #                                  nv_rate_stats_zoom['mean'] - nv_rate_stats_zoom['std'],
    #                                  color='indigo', alpha=0.3)
    #                 ax2.set_xlim(0.95, 1)
    #                 ax2.set_ylim(.2, 1)
    #
    #                 savepath = self.an_path + '/%s/%s/' % (model_name, cv_name)
    #                 if not exists(savepath):
    #                     mkdir(savepath)
    #                 plt.savefig(savepath + 'nv_t_trigger_%i.pdf' % novelty_cls,
    #                             bbox_inches='tight')
    #                 plt.close(fig)

    def plotEffThresholds(self):
        raise NotImplementedError

    def plotDensities(self):
        colors = ['b', 'r', 'g', 'y']
        colors = np.repeat(colors, axis=1)

        class_indices = {value: key for key, value in self.class_labels.items()}
        for nv_cls, modelAnalysis in self.ModelsAnalysis.items():
            i_cls = class_indices[nv_cls]
            colors[i_cls, :] = 'k'
            modelAnalysis.plotDensities(color_matrix = colors)


