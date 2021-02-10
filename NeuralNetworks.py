from termcolor import cprint
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

trainfile='labelFeatureVector.xlsx'




dataset=pd.read_excel(trainfile,names=['constrast', 'homogeneity','energy','correlation','asm',
'dct1','dct2','dct3','dct4','dct5','dct6','dct7','dct8','dct9','dct10',
 'dct11','dct12','dct13','dct14','dct15','dct16','dct17','dct18','dct19','dct20','label'])

dataset=dataset.sort_values('energy', ascending = True)




X_train = dataset.iloc[:70,0:25].values
y_train = dataset.iloc[:70,25].values
X_test=dataset.iloc[:108,0:25].values
y_test=dataset.iloc[:108,25].values

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np_utils.to_categorical(y_train)
encoder = LabelEncoder()
y_test = encoder.fit_transform(y_test)
y_test = np_utils.to_categorical(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = Sequential()
classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 25, units = 40, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'uniform', units =32,  activation = 'sigmoid'))
classifier.add(Dense(kernel_initializer = 'uniform', units = 4,  activation = 'softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=(classifier.fit(X_train,y_train,epochs=100,batch_size=5))
print(len(y_test),len(X_test))

y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred.round(), normalize=True)
classification_report(y_true=y_test,y_pred=y_pred.round())

dogru = 0
yanlis = 0
toplam_veri = len(X_test)

for i in range(toplam_veri):
 zip(y_pred,y_test)
 x=y_pred.round()
 y=y_test
 if y[i][0] == 1.0:
        r = '1'
 elif y[i][1] == 1.0:
        r = '2'
 elif y[i][2] == 1.0:
        r = '3'
 elif y[i][3]== 1.0:
        r = '4'

 if x[i][0] == 1.0:
        p='1'

 elif x[i][1]== 1.0:
        p = '2'
 elif x[i][2] == 1.0:
        p= '3'
 else :
        p ='4'


 if p==r:
        cprint("YSA: " + str(p) + " - Gerçek Hata: " + str(r), "white", "on_green", attrs=['bold'])
        dogru += 1
 else:
        cprint("YSA: " + str(p) + " - Gerçek Hata: " + str(r), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-" * 150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Tespit: ", dogru,
      "\nYanlış tespit: ", yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100 * dogru / toplam_veri)) + "%", sep="")

