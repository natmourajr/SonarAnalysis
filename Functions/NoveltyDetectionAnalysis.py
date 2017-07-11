"""
    This file contents some classification analysis functions
    """

import os
import time
import numpy as np

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.models import load_model

import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('legend',**{'fontsize':15})
plt.rc('font', weight='bold')

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

    def analysis_nu_sweep(self, data, trgt, trn_info=None, min_nu=0.1, max_nu=0.9, nu_step=0.1):
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



