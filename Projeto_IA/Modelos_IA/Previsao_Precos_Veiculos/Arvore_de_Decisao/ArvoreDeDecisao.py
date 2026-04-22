# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics

df = pd.read_csv('vehicle_price_prediction.csv')

# Removendo os atributos do dataset que podem iterferir na previsão.
# Por que remover "make"? "brand_popularity" já é uma representação numérica do valor da marca.
# year e mileage: O dataset já possui vehicle_age e mileage_per_year. Estes são atributos mais informativos para o modelo.
df = df.drop(['make', 'model', 'year', 'mileage', 'trim', 'exterior_color', 'interior_color'], axis=1)

# Transformando atributos de texto para números.
mapa_condition = {'Fair': 1, 'Good': 2, 'Excellent': 3}
df['condition'] = df['condition'].map(mapa_condition)

df['accident_history'] = df['accident_history'].fillna('None')
mapa_accident = {'None': 0, 'Minor': 1, 'Major': 2}
df['accident_history'] = df['accident_history'].map(mapa_accident)

df['seller_type'] = df['seller_type'].map({'Private': 0, 'Dealer': 1})

colunas_nominais = ['transmission', 'fuel_type', 'body_type', 'drivetrain']
df = pd.get_dummies(df, columns=colunas_nominais, drop_first=True)

# Declarando X e Y
X = df.drop(columns=['price'])

y = df['price']

# Separar os dados
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Aplicando o algoritmo
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=10, random_state=0)

# Fazendo o treinamento
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print(y_pred)

# Precisão do modelo com R²
r2 = metrics.r2_score(y_test, y_pred)

print(f"O R-squared (R²) do modelo é: {r2 * 100:.2f}%")

# Métricas
mae = metrics.mean_absolute_error(y_test, y_pred)
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