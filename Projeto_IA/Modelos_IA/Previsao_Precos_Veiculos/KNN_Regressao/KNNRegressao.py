# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('vehicle_price_prediction.csv')

# Tratamento
df1 = df.dropna(subset=['year', 'mileage', 'engine_hp', 'owner_count', 'vehicle_age', 'mileage_per_year', 'brand_popularity', 'price'])

# Declarando X e Y
X = df1[['year', 'mileage', 'engine_hp', 'owner_count', 'vehicle_age', 'mileage_per_year', 'brand_popularity']]
y = df1['price']

# Separar os dados
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = KNeighborsRegressor(n_neighbors=5)

# Treinamento
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Métricas MAE e RMSE
mae = metrics.mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(mae)
print(rmse)


mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

y_mean = float(np.mean(y_test))
mae_pct = (mae / y_mean) *100
rmse_pct = (rmse / y_mean) *100
print('MAE : ', mae_pct)
print('RMSE : ', rmse_pct)