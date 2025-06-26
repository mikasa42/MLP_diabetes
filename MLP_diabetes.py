import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#Monitorar o treinamento
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 1. Carregar o dataset (ex: Pima Indians Diabetes)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
data = pd.read_csv(url, names=col_names)

# 2. Separar features (X) e target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalizar os dados (importante para MLPs!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Criar o modelo MLP
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),  # Camada oculta com 12 neurônios
    Dense(8, activation='relu'),                     # Segunda camada oculta
    Dense(1, activation='sigmoid')                   # Saída binária (0 ou 1)
])
# Largar um neurônio alteatório
model.add(Dropout(0.2))  # Adiciona após uma camada Dense
# 6. Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Treinar o modelo
history = model.fit(X_train, y_train, epochs=120, batch_size=10, validation_split=0.1)

# 8. Avaliar no teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {accuracy:.2f}")

# 9. Fazer uma previsão (exemplo)
sample = X_test[0].reshape(1, -1)  # Pega a primeira amostra do teste
prediction = model.predict(sample)
print(f"Probabilidade de diabetes: {prediction[0][0]:.2f}")

#Visualizar o aprendizado
# 1. Preparar o diretório para logs do TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 2. Treinar o modelo COM os dados corretos (substitua X_train, y_train pelos seus dados)
history = model.fit(
    X_train,  # Dados de treino (features)
    y_train,  # Labels de treino
    epochs=100,
    batch_size=10,
    validation_split=0.1,  # Validação durante o treino
    callbacks=[tensorboard_callback]  # TensorBoard integrado
)

# 3. Avaliar o modelo (opcional)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {accuracy:.2f}")