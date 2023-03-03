
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime, timedelta
import argparse
import time
from time import sleep
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import random
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras import layers,models
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
import shutil
from KpNet_utils import *
from Attention import *
from KpNet_dropout import * 
tfd = tfp.distributions
print_gpu_info = False
scale_value = 10

class KpNetModel:
    model = None
    model_name = None
    callbacks = None
    input = None
    y = None
    X_train=None
    y_train=None
    X_test=None
    y_test=None
    predictions = None
    epistemics = None 
    aleatoric = None
    dates = None 
        
    def __init__(self,model_name='KpNetModel',early_stopping_patience=3):
        self.model_name = model_name
        callbacks = [EarlyStopping(monitor='loss', patience=early_stopping_patience,restore_best_weights=True)]
    if print_gpu_info:
        if tf.test.gpu_device_name() != '/device:GPU:0':
          print('WARNING: GPU device not found.')
        else:
            print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices ) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
            
    def set_data(self,x_tr,y_tr, x_te, y_te):
        self.X_train = x_tr 
        self.y_train = y_tr 
        self.X_test = x_te
        self.y_test = y_te
        self.y = self.y_test
    
    def set_al (self,al):
        self.aleatoric = al 
    
    def set_epis(self,ep):
        self.epistemics = ep
    
    def set_preds (self,p):
        self.predictions = p 
    
    def set_dates(self,d):
        self.dates = d 
            
    def build_base_model(self,
                    input_shape,
                    kl_weight=0.0001,
                    dropout=0.4,
                    bilstm=True,
                    b=2,
                    cnn_only=False):
                input = keras.Input(shape=input_shape)
                self.input = input 
                model = layers.Conv1D(filters=32, kernel_size=1, activation="relu",
                                        name=self.model_name+"_conv",
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                      )(input)
                if not cnn_only:
                    if not bilstm:
                        model =(tf.keras.layers.LSTM(250,
                                                    return_sequences=True,
                                                    name=self.model_name+'_lstm',
                                                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                    bias_regularizer=regularizers.l2(1e-4),
                                                    activity_regularizer=regularizers.l2(1e-5)                                              
                                                     ))(model)
                    else:                            
                        model =tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(250,
                                                    return_sequences=True,
                                                    name=self.model_name+'_lstm',
                                                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                    bias_regularizer=regularizers.l2(1e-4),
                                                    activity_regularizer=regularizers.l2(1e-5)                                              
                                                     ))(model)
                    model = layers.Dropout(dropout)(model)
                    if b == 0:
                        model = (layers.MultiHeadAttention(key_dim=4, num_heads=4, dropout=0,name=self.model_name +'_mh'))(model,model)
                    else:
                        for i in range(b):
                            model = self.transformer_encoder(model,head_size=4,num_heads=4,ff_dim=4,dropout=0, bilstm=bilstm, name_postffix='_enc' + str(i))
                model = layers.Dropout(dropout)(model)
                model = layers.Dense(100,
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                     )(model)
                model = layers.Dropout(dropout)(model)
                model = (tfp.layers.DenseVariational(10, posterior_mean_field, prior_trainable, kl_weight=kl_weight))(model)
                model = (layers.Dropout(dropout))(model)
                model = layers.Dense(1,
                                     activity_regularizer=regularizers.l2(1e-5)
                                     )(model)
                self.model = model 
                return model   
    
    # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      c = np.log(np.expm1(1.))
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t[..., :n],
                         scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
              reinterpreted_batch_ndims=1)),
      ])
      
      # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):
      n = kernel_size + bias_size
      return tf.keras.Sequential([
          tfp.layers.VariableLayer(n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t, scale=1),
              reinterpreted_batch_ndims=1)),
      ])
    def transformer_encoder(self, inputs,head_size=2,num_heads=2,ff_dim=4,dropout=0, bilstm=True, name_postffix='_enc1'):
        x = layers.Dropout(dropout)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
        # x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = x + inputs
        x = layers.Conv1D(filters=ff_dim, kernel_size=inputs.shape[-2], activation="relu")(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        if not bilstm:
            x = (tf.keras.layers.LSTM(250,
                                      return_sequences=True,
                                      name=self.model_name+name_postffix + '_lstm'))(x)               
        else:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(250,
                                         return_sequences=True,
                                         name=self.model_name+name_postffix + '_lstm'))(x)        
        self.model = x 
        return x
    def models(self):
        self.model = models.Model(self.input, self.model)
    def summary(self):
        self.model.summary()
    
    def compile(self,loss='mse',metrics=['mse'], adam_lr=0.0001):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=adam_lr), 
                           loss=loss,
                           metrics=metrics)
    def fit(self,
            X_train, 
            y_train,
            X_valid=None, 
            y_valid=None,
            epochs=100,
            verbose=2,
            batch_size=512):
        validation_data = None 
        if X_valid and y_valid:
            validation_data =[X_valid, y_valid]
        self.model.fit(X_train, 
                       y_train, 
                       epochs=epochs, 
                       verbose=verbose, 
                       batch_size=batch_size,
                       callbacks=self.callbacks)
             
    def uncertainty(self,X_test,y_test, N=100, metric='avg', verbose=1, scale_value=10):
        # predict stochastic dropout model T times
        p_hat = []
        print('\nUncertainty Quantification')
        aleatoric=[]
        epistemic = []
        for t in range(N):
            # preds= self.finetune(y_test, self.model(X_test, training = True), scale_down=True)
            preds= self.model(X_test, training = True)
            preds = preds
            p_hat.append(preds)            
            if verbose == 1 or verbose == True:
                print(self.pad_progress_bar(str(t+1), str(N)) ,' [===== Uncertainty Quantification ======]  - ', self.pad_progress_bar(int(float(((t+1)/N)*100)),N), '%')
        p_hat = np.array(p_hat)
        preds = p_hat 
        # mean prediction
        prediction = np.mean(p_hat, axis=0)/scale_value
        prediction = np.min(p_hat, axis=0)/scale_value
    
        max_p=np.max(preds)
        min_p=np.min(preds)
        preds_min_p= preds-min_p
        max_p_preds = max_p - preds
        multiplication_term = (preds_min_p) * (max_p_preds)
        multiplication_term_mean = np.mean(multiplication_term, axis=0)
        aleatoric=(np.sqrt(multiplication_term_mean)) *2 /scale_value
        epistemic=(np.mean(preds ** 2, axis=0) - np.mean(preds, axis=0) ** 2)
        epistemic = np.sqrt(epistemic*10)/scale_value * 1.2
        
        return np.squeeze(prediction), np.std(np.squeeze(aleatoric)), np.std(np.squeeze(epistemic)), p_hat

    
    def finetune(self,y_test, predictions, metric='process',scale_down=False):
        # if metric =='avg':
        #     return [np.array(p).mean() for p in predictions]
        # if metric == 'max':
        #     return [np.array(p).max() for p in predictions]
        # if metric == 'min':
        #     return [np.array(p).min() for p in predictions]
        results = []
        #print(len(predictions), len(y_test))
        y_test= np.array(y_test)
        for i in range(len(predictions)):
            #print(i)
            t = y_test[i]
            preds = predictions[i]
            results.append(process_val(t, preds,scale_down=scale_down))
        return results
        # return np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic)
        
    def predict(self,X_test,verbose=1):
        predictions = self.model.predict(X_test,
                                         verbose=verbose,
                                         batch_size=len(X_test))
        return np.squeeze(predictions) 
    
    def save_weights(self,num_hours=1,interval_type='hourly', w_dir=None, verbose=True):
        if w_dir is None:
            weight_dir = 'models' 
        else:
            weight_dir = w_dir  
        weight_dir = weight_dir + os.sep + str(num_hours) + interval_type[0]
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
        os.makedirs(weight_dir)
        if verbose:
            log('Saving model weights to directory:', weight_dir)
        self.model.save_weights(weight_dir + os.sep + 'model_weights')
          
    def load_weights(self,num_hours=1,interval_type='hourly',w_dir=None):
        weight_dir = 'models' + os.sep + str(num_hours) + interval_type[0]
        if w_dir is not None:
            weight_dir = w_dir +  os.sep + str(num_hours) + interval_type[0]
        log('Loading weights from model dir:', weight_dir)
        if not os.path.exists(weight_dir):
            weight_dir = 'default_models' +  os.sep + str(num_hours) + interval_type[0]
            print( 'Model weights directory does not exist:', weight_dir, 'trying the default_models directory',weight_dir)
            if not os.path.exists(weight_dir):
                print('Default model does not exist:', weight_dir)
                exit()
        if self.model == None :
            print('You must create and train the model first before loading the weights.')
            exit() 
        self.model.load_weights(weight_dir + os.sep + 'model_weights').expect_partial()   
    
    def load_model(self,input_shape,
                   num_hours,
                    kl_weight=0.0001,
                    dropout=0.4,
                    loss='mse',
                    metrics=['mse'],
                    adam_lr=0.0001,
                    bilstm=True,
                    interval_type='hourly',
                    w_dir=None):
        self.build_base_model(input_shape, kl_weight, dropout,bilstm=bilstm)
        self.models()
        self.compile(loss=loss, metrics=metrics, adam_lr=adam_lr)
        self.load_weights(num_hours, interval_type, w_dir=w_dir)

    def model_weights_exist(self,num_hours=1,interval_type='hourly', w_dir=None,verbose=True) -> bool:
        if w_dir is None:
            weight_dir = 'models' 
        else:
            weight_dir = w_dir  
        weight_dir = weight_dir + os.sep + str(num_hours) + interval_type[0]
        if verbose:
            log('Loading weights from model dir:', weight_dir)
        if not os.path.exists(weight_dir):
            print( 'Model weights directory does not exist:', weight_dir)
            return False
        return True 
                
    def get_model(self):
        return self.model
    def pad_progress_bar(self,n,d):
        n = str(n)
        d = str(d) 
        a = n + '/' + d 
        t = d + '/' + d 
        r = n + '/' + d 
        for c in range(len(t) - len(a)):
            r = ' ' + r
        return r 
        if  len(str(n)) == 5:
            return n + '  '
        if len(str(n)) == 6:
            return n + ' '
        return n
    
    def get_file_name(self,num_hours,n, results_dir='results') -> str:
        return results_dir + os.sep + 'kp_' + str(num_hours)+'h_' + str(n)+'.csv'
        
    def save_results(self,num_hours,results_dir='results'):
        self.y_test = self.y_test/scale_value
        data = {
                'Date': self.dates,
                'Labels':list(self.y_test.flatten()),
                'Predictions':self.predictions,
                'Epistemic':self.epistemics,
                'Aleatoric':self.aleatoric
                }
        df = pd.DataFrame(columns=data.keys())
        df['Date']= self.dates
        df['Labels'] = list(self.y_test.flatten())
        df['Predictions'] = list(self.predictions)
        e = [math.nan for i in range(len(self.predictions))]
        e[0] = self.epistemics
        a = [math.nan for i in range(len(self.predictions))]
        a[0] = self.aleatoric
        df['Epistemic'] = e
        df['Aleatoric'] = a
        log('Saving the result to:', results_dir + os.sep + 'kp_' + str(num_hours)+'h_results.csv',verbose=True)
        df.to_csv(results_dir + os.sep + 'kp_' + str(num_hours)+'h_results.csv',index=None)

    def get_results(self,num_hours,results_dir='results'):
        if not os.path.exists(results_dir + os.sep + 'kp_' + str(num_hours)+'h_results.csv'):
            print('Error: the results file does not exist for h = ' + str(num_hours), 'File must be:', results_dir + os.sep + 'kp_' + str(num_hours)+'h_results.txt')
            exit()
            
        data = pd.read_csv(results_dir + os.sep + 'kp_' + str(num_hours)+'h_results.csv')
        return data['Predictions'].values, (data['Aleatoric'].values)[0],(data['Epistemic'].values)[0], data['Labels'].values, data['Date'].values    
    