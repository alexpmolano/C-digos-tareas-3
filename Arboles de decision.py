#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar las librerías necesarias
import pandas as pd

ruta_archivo = 'C:/Datasets/winequality-red.csv'  
data = pd.read_csv(ruta_archivo)

# Ver las primeras filas del dataset
data.head()


# In[3]:


# Obtener información del dataset (número de registros, tipos de datos, etc.)
print(data.info())

# Revisar si hay valores nulos en las columnas
print("Valores nulos por columna:")
print(data.isnull().sum())


# In[5]:


# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data, palette='viridis')
plt.title('Distribución de la Calidad del Vino')
plt.xlabel('Calidad')
plt.ylabel('Cantidad')
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data, palette='viridis')
plt.title('Distribución de la Calidad del Vino')
plt.xlabel('Calidad')
plt.ylabel('Cantidad')
plt.show()


# In[11]:


X = data.drop('quality', axis=1)  # Características
y = data['quality']  # Variable objetivo


# In[13]:


from sklearn.model_selection import train_test_split

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ver las dimensiones de los conjuntos
print(f'Tamaño del conjunto de entrenamiento: {X_train.shape[0]}')
print(f'Tamaño del conjunto de prueba: {X_test.shape[0]}')


# In[15]:


from sklearn.tree import DecisionTreeClassifier

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)


# In[17]:


from sklearn.metrics import classification_report, confusion_matrix

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el desempeño del modelo
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# In[19]:


import seaborn as sns

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()


# In[21]:


import numpy as np


# In[23]:


import seaborn as sns

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()


# In[25]:


# Importar la librería para el gráfico
importances = model.feature_importances_
feature_names = X.columns

# Crear un DataFrame para las importancias
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Ordenar las características por importancia
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualizar la importancia de las características
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Importancia de las Características en el Modelo')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()


# In[27]:


# Graficar Predicciones vs Realidad
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title('Predicciones vs Realidad')
plt.xlabel('Realidad')
plt.ylabel('Predicciones')
plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Línea de referencia
plt.xlim(0, 10)  # Ajustar los límites según las clases
plt.ylim(0, 10)
plt.grid(True)
plt.show()


# In[29]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Calcular las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Crear un DataFrame para las métricas
metrics_df = pd.DataFrame({
    'Métrica': ['Precisión', 'Recall', 'F1-Score', 'Exactitud'],
    'Valor': [precision, recall, f1, accuracy]
})

# Graficar las métricas
plt.figure(figsize=(10, 6))
sns.barplot(x='Métrica', y='Valor', data=metrics_df, palette='viridis')
plt.title('Métricas de Desempeño del Modelo')
plt.ylim(0, 1)  # Ajustar el límite superior a 1
plt.grid(True)
plt.show()


# In[ ]:




