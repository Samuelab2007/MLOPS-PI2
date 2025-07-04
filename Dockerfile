# Imagen base con Python
FROM python:3.10-slim

# Crear y usar directorio de trabajo
WORKDIR /app

# Copiar requerimientos e instalarlos
COPY requeriments.txt .
RUN pip install --no-cache-dir -r requeriments.txt

# Copiar el archivo principal de la API
COPY scripts/ ./scripts/

# Copiar el modelo
COPY models/test_model.joblib ./models/test_model.joblib

# Exponer el puerto del servidor
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "scripts.main_api:app", "--host", "0.0.0.0", "--port", "8000"]

