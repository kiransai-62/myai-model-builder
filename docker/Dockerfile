FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install framework
RUN pip install --no-cache-dir myai-cli[serve]

# Create a generic entrypoint script
RUN echo '#!/bin/bash\n\
unzip -o /model/myai.myai -d /app/unpacked\n\
cd /app/unpacked\n\
myai serve --host 0.0.0.0 --port 8000' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
