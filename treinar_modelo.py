import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# carregamento o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalização dos dados pros valores de 0 e 1
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# definição do modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),     # camada de entrada imagem 28x28
    Dense(128, activation='relu'),     # camada oculta
    Dense(64, activation='relu'),      # camada oculta
    Dense(10, activation='softmax')    # camada de saída 10 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test)
print(f'\nAcurácia no teste: {acc * 100:.2f}%')

model.save("modelo_digitos.h5")
