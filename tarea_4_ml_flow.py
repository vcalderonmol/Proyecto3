# -*- coding: utf-8 -*-
"""
**4. Tarea 4: Modelamiento**
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor

import mlflow
import mlflow.sklearn

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)

"""### Carga de datos"""

data_df = pd.read_csv("clean_data.csv", sep=',')
data_df.head()

#Quitar variable Unnamed:o
data_df.drop(columns=['Unnamed: 0'], inplace=True)
#Cuenta los nulos
data_df.isnull().sum()

data_df_modeling =data_df.copy()

"""## Tarea 4: Modelamiento Y no escalada

### 4.1 Particionamiento del conjunto de datos en entrenamiento y prueba
"""
#Separar variables indepnedientes y de interes
X = data_df_modeling.drop("PUNT_GLOBAL", axis=1)
Y = data_df_modeling["PUNT_GLOBAL"]
#Separamos las muestras
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Escalamiento
# Inicializar el objeto StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

#Escalamiento en X
#X_train[features_numericas] = scaler_X.fit_transform(X_train[features_numericas])
#X_test[features_numericas] = scaler_X.transform(X_test[features_numericas])

#Adecuamos a numpy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

#Escalamiento en Y
#y_train= scaler_y.fit_transform(y_train)
#y_test = scaler_y.transform(y_test)

#Imprimimos dimensiones
print("Train shape:  ",X_train.shape, y_train.shape)
print("Test shape:  ",X_test.shape, y_test.shape)

#Media y desvest
print("Media de X_train_scaled:", np.mean(X_train, axis=0))
print("Media de y_train_scaled:", np.mean(y_train, axis=0))

# REGISTRAR EL EXPERIMENTO
#mlflow.set_tracking_uri('http://0.0.0.0:5000')
experiment = mlflow.set_experiment("icfes")

"""### 4.2 Entrenamiento de un primer modelo de regresión lineal
"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Linear Regression Model Run"):
  # Crear el modelo
  model = LinearRegression()
  # Entrenar el modelo
  model.fit(X_train, y_train)
  # Obtener las predicciones para el set Tes
  y_pred_test = model.predict(X_test)
  y_test_original = y_test
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = model.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(model, "model-lineal")

"""### 4.3 Entrenamiento de un segundo modelo de regresión lineal (LASSO)"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Lasso Regression Model Run"):
  # Crear el modelo
  model_lasso = Lasso(alpha=0.01)
  # Entrenar el modelo
  model_lasso.fit(X_train, y_train)
  # Obtener las predicciones para el set Test
  y_pred_test = model_lasso.predict(X_test)
  y_test_original = y_test
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = model_lasso.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(model_lasso, "model-lasso")

# Initialize an empty list to store selected features
selected_features = []
selected_features_index = []
feature_names = X.columns
for i, feature in enumerate(feature_names):
  if abs(model_lasso.coef_[i]) > 0.0001:
    selected_features.append(feature)
    selected_features_index.append(i)

"""### 4.4 Entrenamiento de un tercer modelo de regresión lineal (BAGGING)"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="BAGGING Regression Model Run"):
  # Crear el modelo base (Lasso)
  model = LinearRegression()
  # Crear el modelo Bagging con Lasso como estimador base
  bagging_lasso = BaggingRegressor(model, n_estimators=15, random_state=42)
  # Entrenar el modelo Bagging
  bagging_lasso.fit(X_train[:, selected_features_index], y_train)
  # Obtener las predicciones para el set Test
  y_pred_test = bagging_lasso.predict(X_test[:, selected_features_index])
  y_test_original = y_test
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = bagging_lasso.score(X_test[:, selected_features_index],y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(bagging_lasso, "model-bagging")


"""## Tarea 4: Modelamiento Y escalada
### 4.1 Particionamiento del conjunto de datos en entrenamiento y prueba
"""
#Separar variables indepnedientes y de interes
X = data_df_modeling.drop("PUNT_GLOBAL", axis=1)
Y = data_df_modeling["PUNT_GLOBAL"]

#Separamos las muestras
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Escalamiento
# Inicializar el objeto StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

#Adecuamos a numpy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)

#Escalamiento en Y
y_train= scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

#Imprimimos dimensiones
print("Train shape:  ",X_train.shape, y_train.shape)
print("Test shape:  ",X_test.shape, y_test.shape)

#Media y desvest
print("Media de X_train_scaled:", np.mean(X_train, axis=0))
print("Media de y_train_scaled:", np.mean(y_train, axis=0))

"""### 4.2 Entrenamiento de un primer modelo de regresión lineal
"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Escalado Linear Regression Model Run"):
  # Crear el modelo
  model = LinearRegression()
  # Entrenar el modelo
  model.fit(X_train, y_train)
  # Obtener las predicciones para el set Tes
  y_pred_test = model.predict(X_test)
  y_test_original = y_test
  # Desescalar
  y_pred_test = scaler_y.inverse_transform(y_pred_test)
  y_test_original = scaler_y.inverse_transform(y_test)
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = model.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(model, "model-lineal-scaled")

"""### 4.3 Entrenamiento de un segundo modelo de regresión lineal (LASSO)"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Escalado Lasso Regression Model Run"):
  # Crear el modelo
  model_lasso = Lasso(alpha=0.01)
  # Entrenar el modelo
  model_lasso.fit(X_train, y_train)
  # Obtener las predicciones para el set Test
  y_pred_test = model_lasso.predict(X_test)
  y_test_original = y_test
  # Desescalar
  y_pred_test = scaler_y.inverse_transform(y_pred_test)
  y_test_original = scaler_y.inverse_transform(y_test)
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = model_lasso.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(model_lasso, "model-lasso-scaled")

"""### 4.4 Entrenamiento de un tercer modelo de regresión lineal (BAGGING)"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Escalado BAGGING Regression Model Run"):
  # Crear el modelo base (Lasso)
  model = LinearRegression()
  # Crear el modelo Bagging con Lasso como estimador base
  bagging_lasso = BaggingRegressor(model, n_estimators=15, random_state=42)
  # Entrenar el modelo Bagging
  bagging_lasso.fit(X_train, y_train)
  # Obtener las predicciones para el set Test
  y_pred_test = bagging_lasso.predict(X_test)
  y_test_original = y_test
  # Desescalar
  y_pred_test = scaler_y.inverse_transform(y_pred_test)
  y_test_original = scaler_y.inverse_transform(y_test)
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = bagging_lasso.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(bagging_lasso, "model-bagging-scaled")

"""## Tarea 4: Una primera versión de Modelo """
data_df = pd.read_csv("first_clean_data.csv", sep=',')
data_df.drop(columns=['Unnamed: 0'], inplace=True)
data_df_modeling =data_df.copy()

"""### 4.1 Particionamiento del conjunto de datos en entrenamiento y prueba"""
#Separar variables indepnedientes y de interes
X = data_df_modeling.drop("PUNT_GLOBAL", axis=1)
Y = data_df_modeling["PUNT_GLOBAL"]

#Separamos las muestras
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Escalamiento
# Inicializar el objeto StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

#Adecuamos a numpy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


#Imprimimos dimensiones
print("Train shape:  ",X_train.shape, y_train.shape)
print("Test shape:  ",X_test.shape, y_test.shape)

#Media y desvest
print("Media de X_train_scaled:", np.mean(X_train, axis=0))
print("Media de y_train_scaled:", np.mean(y_train, axis=0))

"""### 4.2 Entrenamiento de un primer modelo de regresión lineal"""
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="1st Linear Regression Model Run"):
  # Crear el modelo
  model = LinearRegression()
  # Entrenar el modelo
  model.fit(X_train, y_train)
  # Obtener las predicciones para el set Tes
  y_pred_test = model.predict(X_test)
  y_test_original = y_test
  # Calcular métricas
  # MAE
  mae = mean_absolute_error(y_test_original, y_pred_test)
  #R2
  r2 = model.score(X_test,y_test)
  #Registramos los valores
  mlflow.log_metric("mae", mae)
  mlflow.log_metric("r2", r2)
  #Guardar el modelo
  mlflow.sklearn.log_model(model, "model-lineal-1st")


