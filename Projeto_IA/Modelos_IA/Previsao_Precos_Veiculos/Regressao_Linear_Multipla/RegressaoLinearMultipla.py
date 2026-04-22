# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('vehicle_price_prediction.csv')

# Tratamento
df1 = df.dropna(subset=['year', 'mileage', 'engine_hp', 'owner_count', 'vehicle_age', 
                        'mileage_per_year', 'brand_popularity', 'price'])

# Declarar X e Y
x = df1[['year', 'mileage', 'engine_hp', 'owner_count', 'vehicle_age', 'mileage_per_year', 'brand_popularity']]
y = df1['price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state = 0
)

# Treinamento
modelopreditor = LinearRegression()
modelopreditor.fit(x_train, y_train)

# Coeficientes
coeff = pd.DataFrame(modelopreditor.coef_, x.columns, columns=['Coeficiente'])
print(coeff)

# Intercepto (constante da equação)
b0 = modelopreditor.intercept_

# Lista de coeficientes
b = modelopreditor.coef_

# Comparação das previsões com os valores reais do dataset
predictions = modelopreditor.predict(x_test)
comparacao = pd.DataFrame({
    'Previsto': predictions,
    'Real': y_test.values
})
print(comparacao.head())

# Métricas 
print("MAE: ",metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

mae = metrics.mean_absolute_error(
    y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

y_mean = float(np.mean(y_test))
mae_pct = (mae / y_mean) *100
rmse_pct = (rmse / y_mean) *100
print('MAE : ', mae_pct)
print('RMSE : ', rmse_pct)