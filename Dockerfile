FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy framework
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Install framework
RUN pip install --no-cache-dir -e ".[models,train,retrieval,serve]"

# Copy user project (optional: mount at runtime)
COPY fittrack-ai/ /project/

WORKDIR /project

# Expose port
EXPOSE 8000

# Start server
CMD ["myai", "serve", "--host", "0.0.0.0", "--port", "8000"]
