import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("Churn_Modelling.csv")
data.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)
# print(data)
x=data.drop(columns=['Exited'])
y=data['Exited']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42) 

transformar = ColumnTransformer(transformers=[
    ('geo',OneHotEncoder(sparse_output=False,drop='first'),['Geography','Gender'])
],remainder='passthrough')

x_train_transformed=transformar.fit_transform(x_train)
x_test_transformed=transformar.transform(x_test)


scaler = StandardScaler()
scaler.fit_transform(x_train_transformed)
scaler.transform(x_test_transformed)

model = Sequential()

model.add(Dense(16, activation='relu', input_dim=11))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
result = model.fit(x_train_transformed,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)

y_pred = model.predict(x_test_transformed)
