# Reconhecimento de Dígitos Numéricos com Keras

Projeto desenvolvido para a disciplina de Inteligência Artificial com foco em **Deep Learning**, utilizando **Keras** e a base de dados **MNIST** para realizar o reconhecimento de dígitos numéricos manuscritos (0 a 9).

##  Integrantes
- Gabriel Mazzotti
- Isabella Mindof

##  Estrutura do Projeto
- ├── treinar_modelo.py # Script para treinar o modelo
- ├── reconhecer_digito.py # Script para testar o reconhecimento
- ├── modelo_digitos.h5 # Modelo treinado pronto para uso

##  Funcionamento do Código

### `treinar_modelo.py`
Este script realiza:
- Carregamento da base MNIST (imagens de 28x28 pixels de dígitos 0–9)
- Pré-processamento (normalização e one-hot encoding)
- Definição e treinamento de um modelo sequencial em Keras
- Avaliação do desempenho em dados de teste
- Salvamento do modelo treinado no arquivo `modelo_digitos.h5`

### `reconhecer_digito.py`
Este script:
- Carrega o modelo treinado salvo (`modelo_digitos.h5`)
- Usa o conjunto de teste da base MNIST
- Mostra uma imagem de teste e imprime:
- O rótulo real (número correto)
- O número previsto pelo modelo
- Você pode escolher qual imagem testar modificando a variável index no código reconhcer_digito.py.
 
  ---

##  Como Executar

### 🔧 Requisitos

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib (para visualizar a imagem)

### Instalar dependências

```bash
pip install tensorflow numpy matplotlib

## Fontes

- https://github.com/ProfessorFilipo/PythonAI/tree/main/DeepLearning
- https://www.w3schools.com/python/matplotlib_intro.asp
- https://www.tensorflow.org/tutorials/keras/classification?hl=pt-br


