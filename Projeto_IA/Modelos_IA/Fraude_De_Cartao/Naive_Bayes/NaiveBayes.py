# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Fazer upload manual do arquivo
from google.colab import files
import io
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]), on_bad_lines='skip')

# Tratamento
df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'first', 'last','street', 
                 'city', 'state', 'dob', 'trans_num'], inplace=True, errors='ignore')
df.dropna(inplace=True)

for col in ['merchant', 'category', 'gender', 'job']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Declarar X e Y
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Treinamento
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Matriz de Confusão
y_pred = nb_model.predict(X_test)

print("\n Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Exemplo
example = X_test[0].reshape(1, -1)
pred = nb_model.predict(example)
print(f"\n🔍 Exemplo de previsão: {'Fraude' if pred[0] == 1 else 'Transação normal'}")