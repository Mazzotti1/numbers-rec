import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# carregar o modelo que foi treinado
model = load_model("modelo_digitos.h5")

# carregar dados de teste do MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255.0

# selecionar uma imagem de exemplo
index = 0  # esse valor deifine o numero desenhado do modelo q foi treinado
imagem = x_test[index]
rotulo_esperado = y_test[index]

for i in range(10):
    print(f"Índice {i} → Número desenhado: {y_test[i]}")


# mostra a imagem com o matploit
plt.imshow(imagem, cmap='gray')
plt.title("Imagem de Teste")
plt.show()

imagem_input = np.expand_dims(imagem, axis=0)  # dimensões do batch

predicao = model.predict(imagem_input)
classe_predita = np.argmax(predicao)

print(f"Valor real: {rotulo_esperado}")
print(f"Valor predito: {classe_predita}")
