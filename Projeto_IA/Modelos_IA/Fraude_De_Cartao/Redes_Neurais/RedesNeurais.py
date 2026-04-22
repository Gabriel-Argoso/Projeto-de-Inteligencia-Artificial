# Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fazer upload manual do arquivo
from google.colab import files
uploaded = files.upload()

# Verificar nome do arquivo enviado
import io
df = pd.read_csv(io.BytesIO(uploaded['fraud test.csv']), on_bad_lines='skip')

file_name = list(uploaded.keys())[0]  # nome do arquivo enviado
df = pd.read_csv(io.BytesIO(uploaded[file_name]), on_bad_lines='skip')

# Tratamento
df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 
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
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Matriz de Confusão
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred))