#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar las librerías necesarias
import pandas as pd


ruta_archivo = 'C:/Datasets/heart_cleveland_upload.csv'  
data = pd.read_csv(ruta_archivo)

# Ver las primeras filas del dataset
data.head()


# In[3]:


# Resumen general del dataset
data.info()

# Verificar valores nulos en el dataset
print("Valores nulos por columna:")
print(data.isnull().sum())

# Estadísticas descriptivas para las columnas numéricas
data.describe()


# In[5]:


# Importar las librerías necesarias
from sklearn.model_selection import train_test_split

# Separar las características (X) y la variable objetivo (y)
X = data.drop('condition', axis=1)  # Todas las columnas excepto la variable objetivo
y = data['condition']  # La variable objetivo

# Dividir el dataset en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar el tamaño de los datos de entrenamiento y prueba
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")


# In[7]:


# Importar la librería de Regresión Logística
from sklearn.linear_model import LogisticRegression

# Crear el modelo de Regresión Logística
logreg = LogisticRegression(max_iter=1000)

# Entrenar el modelo con los datos de entrenamiento
logreg.fit(X_train, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = logreg.predict(X_test)

# Mostrar las primeras 10 predicciones
print("Primeras 10 predicciones:", y_pred[:10])


# In[9]:


# Importar las métricas necesarias
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mostrar los resultados
print(f"\nExactitud (Accuracy): {accuracy}")
print(f"Precisión (Precision): {precision}")
print(f"Sensibilidad (Recall): {recall}")
print(f"F1 Score: {f1}")


# In[11]:


# Importar las librerías para graficar
import matplotlib.pyplot as plt
import seaborn as sns

# Graficar la matriz de confusión
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusión")
plt.ylabel("Valor Real")
plt.xlabel("Predicción")
plt.show()


# In[13]:


# Importar las métricas necesarias para la curva ROC
from sklearn.metrics import roc_curve, roc_auc_score

# Calcular la probabilidad de predicción
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calcular el AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"Área Bajo la Curva (AUC): {auc}")

# Graficar la curva ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()


# In[ ]:




