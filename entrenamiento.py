import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Carga de datos
# Asumiendo que los archivos están en el directorio actual
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

def prepare_data(df):
    # La primera columna es el label, las demás son los 784 píxeles
    y = df['label'].values
    x = df.drop('label', axis=1).values
    
    # Reshape: De vector (784,) a matriz (28, 28, 1)
    # Normalización: Dividimos por 255 para llevar el rango a [0, 1]
    x = x.reshape(-1, 28, 28, 1) / 255.0
    return x, y

# Verificación de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configuración para que no reserve toda la memoria VRAM de golpe
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detectada: {gpus}")
        BATCH_SIZE = 128  # Aprovechamos la VRAM para procesar más imágenes a la vez
        EPOCHS = 32       # Podemos permitirnos más épocas en menos tiempo
    except RuntimeError as e:
        BATCH_SIZE = 32
        EPOCHS = 8
        print(e)
else:
    BATCH_SIZE = 32
    EPOCHS = 8    
    print("No se detectó GPU. Se usará la CPU.")

x_train, y_train = prepare_data(train_df)
x_test, y_test = prepare_data(test_df)

# 2. Definición del modelo (Arquitectura CNN)
model = models.Sequential([
    # Bloque 1: Extracción de bordes y formas básicas
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Bloque 2: Características más complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),
    
    # Bloque 3: Clasificador final
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    # Usamos 25 porque aunque faltan letras, los IDs llegan hasta el 24
    layers.Dense(25, activation='softmax') 
])

# 3. Compilación
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenamiento
print("Iniciando entrenamiento...")
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)
# Guardar el modelo completo en formato nativo de Keras (.keras)
model.save('modelo_sign_language.keras')
print("Modelo guardado exitosamente como 'modelo_sign_language.keras'")

# 5. Evaluación final
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nAccuracy en el set de prueba: {test_acc*100:.2f}%')