FROM python:3.10-slim

WORKDIR /app

# Copia i file requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dell'applicazione
COPY . .

# Esponi la porta per FastAPI
EXPOSE 8080

# Comando per avviare FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]