ARG PYTHON_VERSION=3.9.17

FROM python:3.9.17
# FROM registry-docker-registry.registry.svc.cluster.local:5001/python:3.9.17
# Establece el directorio de trabajo en /app
WORKDIR /app 

# Crear un archivo temporal para las dependencias
# COPY requirements.txt .
# COPY ./requirements.txt /app/requirements.txt

# Copiar los archivos de artefactos
COPY . /app/

# ARG APT_PACKAGES
# RUN apt-get update && apt-get install -y --no-install-recommends ${APT_PACKAGES} && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instala las dependencias especificadas en el archivo temporal
RUN pip install --upgrade pip setuptools && pip install --no-cache-dir --ignore-installed -r requirements.txt

ENTRYPOINT ["python", "app.py"]