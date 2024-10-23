#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


# Cargar el dataset 
ruta = 'C:/Datasets/CAR_DETAILS_FROM_CAR_DEKHO.csv'  # Ruta completa o relativa
data = pd.read_csv(ruta)

# Ver las primeras filas del dataset
data.head()


# In[5]:


# Información general del dataset
data.info()

# Ver si hay valores nulos
data.isnull().sum()

# Estadísticas descriptivas del dataset
data.describe()


# In[11]:


# Atributos (características) y variable objetivo
X = data[['name', 'year','km_driven', 'fuel','seller_type','owner',  ]] 
y = data['selling_price']  # Precio del automóvil (variable objetivo)


# In[13]:


# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[17]:


# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)


# In[19]:


# Convertir las variables categóricas en variables dummy
data_encoded = pd.get_dummies(data, drop_first=True)

# Ver las primeras filas del dataset codificado
data_encoded.head()


# In[21]:


X = data_encoded.drop('selling_price', axis=1) 
y = data_encoded['selling_price']


# In[23]:


# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)


# In[27]:


# Realizar predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio
r2 = r2_score(y_test, y_pred)  # Coeficiente de determinación R^2

print(f"Error Cuadrático Medio: {mse}")
print(f"R^2 Score: {r2}")


# In[ ]:




