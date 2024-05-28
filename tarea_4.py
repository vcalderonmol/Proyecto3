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

#Normalizar variables númericas: features_numericas
#features_numericas_wiyhout_y= features_numericas

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
experiment = mlflow.set_experiment("icfes-no-escalado")


"""### 4.2 Entrenamiento de un primer modelo de regresión lineal
"""
with mlflow.start_run(experiment_id=experiment.experiment_id):
  # Crear el modelo
  model = LinearRegression()
  # Entrenar el modelo
  model.fit(X_train, y_train)
coeficientes=model.coef_[0]
features = X.columns
for i, feature in enumerate(features):
  print(f"Coeficiente de {feature}: {coeficientes[i]}")
print(f"\nIntercepto: {model.intercept_[0]}")

"""**Validación Cuantitativa**"""
# Obtener las predicciones para el set Train
y_pred = model.predict(X_train)
y_train_original = y_train
# MAE
mae = mean_absolute_error(y_train_original, y_pred)
# MSE
mse = mean_squared_error(y_train_original, y_pred)
# RMSE
rmse = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", model.score(X_train,y_train))
# Obtener las predicciones para el set Tes
y_pred_test = model.predict(X_test)
y_test_original = y_test
# MAE
mae = mean_absolute_error(y_test_original, y_pred_test)
# MSE
mse = mean_squared_error(y_test_original, y_pred_test)
# RMSE
rmse = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", model.score(X_test,y_test))


"""### 4.3 Entrenamiento de un segundo modelo de regresión lineal (LASSO)"""
# Crear el modelo
model_lasso = Lasso(alpha=0.01)
# Entrenar el modelo
model_lasso.fit(X_train, y_train)
# Initialize an empty list to store selected features
selected_features = []
selected_features_index = []
feature_names = X.columns
for i, feature in enumerate(feature_names):
  if abs(model_lasso.coef_[i]) > 0.0001:
    selected_features.append(feature)
    selected_features_index.append(i)
    print(f"Coeficiente de {feature}: {model_lasso.coef_[i]}")

print("Intercept: ", model_lasso.intercept_[0])

"""**Validación Cuantitativa**"""

# Obtener las predicciones para el set Train
y_pred_lasso = model_lasso.predict(X_train).reshape(-1,1)
y_train_original = y_train
# MAE
mae_lasso  = mean_absolute_error(y_train_original, y_pred_lasso)
# MSE
mse_lasso  = mean_squared_error(y_train_original, y_pred_lasso)
# RMSE
rmse_lasso  = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae_lasso)
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R2:", model_lasso.score(X_train,y_train))

# Obtener las predicciones para el set Test
y_pred_test_lasso = model_lasso.predict(X_test)
y_test_original = y_test
# MAE
mae_lasso = mean_absolute_error(y_test_original, y_pred_test_lasso)
# MSE
mse_lasso = mean_squared_error(y_test_original, y_pred_test_lasso)
# RMSE
rmse_lasso = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae_lasso)
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R2:", model_lasso.score(X_test,y_test))

"""### 4.4 Entrenamiento de un tercer modelo de regresión lineal (BAGGING)"""

from sklearn.ensemble import BaggingRegressor

# Crear el modelo base (Lasso)
model = LinearRegression()

# Crear el modelo Bagging con Lasso como estimador base
bagging_lasso = BaggingRegressor(model, n_estimators=15, random_state=42)

# Entrenar el modelo Bagging
bagging_lasso.fit(X_train[:, selected_features_index], y_train)

"""**Validación Cuantitativa**"""

# Obtener las predicciones para el conjunto de entrenamiento
y_pred_train_bagging = bagging_lasso.predict(X_train[:, selected_features_index])
y_train_original = y_train

# Calcular las métricas de evaluación para el conjunto de entrenamiento
mae_train_bagging = mean_absolute_error(y_train_original, y_pred_train_bagging)
mse_train_bagging = mean_squared_error(y_train_original, y_pred_train_bagging)
rmse_train_bagging = np.sqrt(mse_train_bagging)
r2_train_bagging = bagging_lasso.score(X_train[:, selected_features_index], y_train)


# Imprimir las métricas de evaluación
print("Métricas de evaluación para el conjunto de entrenamiento:")
print("MAE:", mae_train_bagging)
print("MSE:", mse_train_bagging)
print("RMSE:", rmse_train_bagging)
print("R2:", r2_train_bagging)

# Obtener las predicciones para el conjunto de prueba
y_pred_test_bagging = bagging_lasso.predict(X_test[:, selected_features_index])
y_test_original = y_test

# Calcular las métricas de evaluación para el conjunto de prueba
mae_test_bagging = mean_absolute_error(y_test_original, y_pred_test_bagging)
mse_test_bagging = mean_squared_error(y_test, y_pred_test_bagging)
rmse_test_bagging = np.sqrt(mse_test_bagging)
r2_test_bagging = bagging_lasso.score(X_test[:, selected_features_index], y_test)

# Imprimir los resultados
print("Métricas de evaluación para el conjunto de prueba:")
print("MAE:", mae_test_bagging)
print("MSE:", mse_test_bagging)
print("RMSE:", rmse_test_bagging)
print("R2:", r2_test_bagging)

"""## Tarea 4: Modelamiento Y escalada
### 4.1 Particionamiento del conjunto de datos en entrenamiento y prueba
"""

#Separar variables indepnedientes y de interes
X = data_df_modeling.drop("PUNT_GLOBAL", axis=1)
Y = data_df_modeling["PUNT_GLOBAL"]

#Normalizar variables númericas: features_numericas
#features_numericas_wiyhout_y= features_numericas

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

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

coeficientes=model.coef_[0]
features = X.columns

for i, feature in enumerate(features):
  print(f"Coeficiente de {feature}: {coeficientes[i]}")

print(f"\nIntercepto: {model.intercept_[0]}")

"""**Validación Cuantitativa**"""

# Obtener las predicciones para el set Train
y_pred = model.predict(X_train)
y_train_original = y_train

# MAE
mae = mean_absolute_error(y_train_original, y_pred)
# MSE
mse = mean_squared_error(y_train_original, y_pred)
# RMSE
rmse = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", model.score(X_train,y_train))

# Obtener las predicciones para el set Tes
y_pred_test = model.predict(X_test)
y_test_original = y_test

# MAE
mae = mean_absolute_error(y_test_original, y_pred_test)
# MSE
mse = mean_squared_error(y_test_original, y_pred_test)
# RMSE
rmse = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", model.score(X_test,y_test))

"""### 4.3 Entrenamiento de un segundo modelo de regresión lineal (LASSO)"""

# Crear el modelo
model_lasso = Lasso(alpha=0.01)

# Entrenar el modelo
model_lasso.fit(X_train, y_train)

# Initialize an empty list to store selected features
selected_features = []
selected_features_index = []
feature_names = X.columns
for i, feature in enumerate(feature_names):
  if abs(model_lasso.coef_[i]) > 0.0001:
    selected_features.append(feature)
    selected_features_index.append(i)
    print(f"Coeficiente de {feature}: {model_lasso.coef_[i]}")

print("Intercept: ", model_lasso.intercept_[0])

"""**Validación Cuantitativa**"""

# Obtener las predicciones para el set Train
y_pred_lasso = model_lasso.predict(X_train).reshape(-1,1)
y_train_original = y_train
# MAE
mae_lasso  = mean_absolute_error(y_train_original, y_pred_lasso)
# MSE
mse_lasso  = mean_squared_error(y_train_original, y_pred_lasso)
# RMSE
rmse_lasso  = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae_lasso)
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R2:", model_lasso.score(X_train,y_train))

# Obtener las predicciones para el set Test
y_pred_test_lasso = model_lasso.predict(X_test)
y_test_original = y_test
# MAE
mae_lasso = mean_absolute_error(y_test_original, y_pred_test_lasso)
# MSE
mse_lasso = mean_squared_error(y_test_original, y_pred_test_lasso)
# RMSE
rmse_lasso = np.sqrt(mse)
# Imprimir los resultados
print("MAE:", mae_lasso)
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R2:", model_lasso.score(X_test,y_test))

"""### 4.4 Entrenamiento de un tercer modelo de regresión lineal (BAGGING)"""

from sklearn.ensemble import BaggingRegressor

# Crear el modelo base (Lasso)
model = LinearRegression()

# Crear el modelo Bagging con Lasso como estimador base
bagging_lasso = BaggingRegressor(model, n_estimators=15, random_state=42)

# Entrenar el modelo Bagging
bagging_lasso.fit(X_train[:, selected_features_index], y_train)

"""**Validación Cuantitativa**"""

# Obtener las predicciones para el conjunto de entrenamiento
y_pred_train_bagging = bagging_lasso.predict(X_train[:, selected_features_index])
y_train_original = y_train

# Calcular las métricas de evaluación para el conjunto de entrenamiento
mae_train_bagging = mean_absolute_error(y_train_original, y_pred_train_bagging)
mse_train_bagging = mean_squared_error(y_train_original, y_pred_train_bagging)
rmse_train_bagging = np.sqrt(mse_train_bagging)
r2_train_bagging = bagging_lasso.score(X_train[:, selected_features_index], y_train)


# Imprimir las métricas de evaluación
print("Métricas de evaluación para el conjunto de entrenamiento:")
print("MAE:", mae_train_bagging)
print("MSE:", mse_train_bagging)
print("RMSE:", rmse_train_bagging)
print("R2:", r2_train_bagging)

# Obtener las predicciones para el conjunto de prueba
y_pred_test_bagging = bagging_lasso.predict(X_test[:, selected_features_index])
y_test_original = y_test

# Calcular las métricas de evaluación para el conjunto de prueba
mae_test_bagging = mean_absolute_error(y_test_original, y_pred_test_bagging)
mse_test_bagging = mean_squared_error(y_test, y_pred_test_bagging)
rmse_test_bagging = np.sqrt(mse_test_bagging)
r2_test_bagging = bagging_lasso.score(X_test[:, selected_features_index], y_test)

# Imprimir los resultados
print("Métricas de evaluación para el conjunto de prueba:")
print("MAE:", mae_test_bagging)
print("MSE:", mse_test_bagging)
print("RMSE:", rmse_test_bagging)
print("R2:", r2_test_bagging)