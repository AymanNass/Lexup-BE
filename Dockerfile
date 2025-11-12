
FROM python:3.10-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Imposta una variabile PORT predefinita.
# Cloud Run la sovrascriverà, ma è utile per test locali.
ENV PORT 8080

# Copia i file requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dell'applicazione
COPY . .

# La riga EXPOSE è solo documentazione, ma è buona pratica
EXPOSE 8080

# Comando per avviare FastAPI con Uvicorn USANDO la variabile $PORT
# Cloud Run sostituirà ${PORT} con il valore corretto (es. 8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
