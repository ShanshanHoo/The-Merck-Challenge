# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:49:43 2020

@author: 31509
"""
import pandas as pd
from pandas import DataFrame

file_path = r"E:\\pydoc\\QSAR_data.txt"                            ## 文档路径（改）

def get_line_context(file_path, line_number):
    import linecache
    return linecache.getline(file_path, line_number).strip()

X_train = DataFrame()
X_test = DataFrame()
y_train = DataFrame()
y_test = DataFrame()



for i in range(11,15):
    l = get_line_context(file_path, 2*(i+1)-1)
    x = pd.read_csv(l)
    row = round(x.shape[0]*0.75)
    train = x.iloc[:row,:]
    test = x.iloc[row:,:]
    X_train = pd.concat([X_train,train],axis = 0)
    X_test = pd.concat([X_test,test],axis = 0)
    del  train, test
    l = get_line_context(file_path,2*(i+1))
    y = pd.read_csv(l)
    train = y.iloc[:row,:]
    test = y.iloc[row:,:]
    y_train = pd.concat([y_train,train],axis=0)
    y_test = pd.concat([y_test,test],axis=0)
    del row, y, train, test

X_train.to_csv('MDNN_xtrain.csv')
y_train.to_csv('MDNN_ytrain.csv')

X_test.to_csv('MDNN_xtest.csv')
y_test.to_csv('MDNN_ytest.csv')


###Joint DNNs

## time record
from timeit import default_timer as timer
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

def coeff_determination(y_test, y_pred):
     from keras import backend as K
     SS_res = (y_test-y_pred)*(y_test-y_pred)
     SS_res = SS_res.sum()
     SS_tot = (y_test - y_test.mean())*(y_test - y_test.mean())
     SS_tot = SS_tot.sum()
     R_2 =  1 - SS_res/(SS_tot + K.epsilon())
     return R_2
 
result =  pd.DataFrame(index = [1],columns=['JDNN','JDNN_RP'])

# Initialising the DNN without RP
batch = 200
epo = 350
model = Sequential()
model.optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=False)
model.add(Dense(4000, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(units = 2000, activation = 'relu'))
model.add(Dense(units = 1000, activation = 'relu'))
model.add(Dense(units = 1000, activation = 'relu'))
model.add(Dense(units = 15))
model.compile(optimizer = 'SGD', loss = 'mean_squared_error')
# Fitting the DNN to the Training set
model.fit(X_train, y_train, batch_size = batch, epochs = epo,callbacks=[cb])
y_pred = model.predict(X_test)
r_square = coeff_determination(y_test,y_pred)
result.iloc[0,0] = r_square
model.save(
    "E:\\pydoc",                                             ##模型保存路径（改）
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)

# DNN with RP
import numpy as np
from sklearn.random_projection import SparseRandomProjection
transformer = SparseRandomProjection(random_state=0)
Xtrain_rp = transformer.fit_transform(X_train)
Xtest_rp = transformer.fit_transform(X_test)

batch = 200
epo = 350
modelrp = Sequential()
modelrp.optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=False)
modelrp.add(Dense(4000, activation = 'relu', input_dim = X_train.shape[1]))
modelrp.add(Dense(units = 2000, activation = 'relu'))
modelrp.add(Dense(units = 1000, activation = 'relu'))
modelrp.add(Dense(units = 1000, activation = 'relu'))
modelrp.add(Dense(units = 15))
modelrp.compile(optimizer = 'SGD', loss = 'mean_squared_error')
# Fitting the DNN to the Training set
modelrp.fit(Xtrain_rp, y_train, batch_size = batch, epochs = epo,callbacks=[cb])
y_pred = modelrp.predict(Xtest_rp)
r_square = coeff_determination(y_test,y_pred)
result.iloc[0,1] = r_square
modelrp.save(
    "E:\\pydoc",                                                 ##模型保存路径（改）
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)