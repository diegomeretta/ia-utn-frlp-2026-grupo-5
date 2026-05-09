# Imagen base oficial de TensorFlow con soporte GPU
FROM tensorflow/tensorflow:latest-gpu

# Evita que Python genere archivos .pyc y permite logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiamos los archivos de dependencias
# Nota: Como usas Pipenv, exportaremos a requirements.txt o instalamos pipenv
RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock ./

# Instalamos las dependencias en el sistema del contenedor (sin entorno virtual separado)
RUN pipenv install --system --deploy --ignore-pipfile

# Copiamos el resto del código
COPY . .

# Comando por defecto
CMD ["python", "main.py"]