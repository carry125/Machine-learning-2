# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:02:39 2021

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#把資料缺失過多的數據切除
#日期不重要，地點與風向可以透過其他數值資料進行演算法歸納，
#而今天是否下雨與明天是否下雨邏輯上關聯度太低(應該要依靠溼度等數值)
dataset=pd.read_csv('final_project_dataset_2.csv')
dataset.drop(["Date", "Location","WindDir9am","WindDir3pm", "WindGustDir", "RainToday"], axis = 1, inplace = True)
dataset.dropna(inplace=True)
dataset.RainTomorrow = [1 if each == "Yes" else 0 for each in dataset.RainTomorrow]
x= dataset.iloc[:,:-1].values#除了最後的所有DATA
y = dataset.iloc[:,[16]].values#最後的DATA


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

#使用PCA過濾大量資料
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
#預測測試項
y_pred=classifier.predict(X_test)

#做出confusion matrix
from sklearn.metrics import confusion_matrix
cm_RF=confusion_matrix(y_test, y_pred)

cm_RFPCA=confusion_matrix(y_test, y_pred)

