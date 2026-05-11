FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time (faster cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app files
COPY data/ ./data/
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
