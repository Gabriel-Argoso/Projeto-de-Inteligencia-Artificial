# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("vehicle_price_prediction.csv")

# Tratamento
df = df.drop(["model", "trim", "interior_color"], axis=1, errors='ignore')
# Pegando uma amostra menor pra rodar mais rápido
df = df.sample(frac=0.02, random_state=1)

# Declarando X e Y
X = df.drop("price", axis=1)
y = df["price"]

# Separando colunas numéricas e categóricas
num = X.select_dtypes(include=["int64", "float64"])
cat = X.select_dtypes(include=["object"])

# Convertendo as categóricas
enc = OneHotEncoder(handle_unknown="ignore")
cat_enc = enc.fit_transform(cat).toarray()

X_all = np.concatenate([num.values, cat_enc], axis=1)

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Treinamento
model = SVR()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# Métrica MAE
print("\nResultados do modelo:")
print("MAE:", mean_absolute_error(y_test, pred))