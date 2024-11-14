# This builds the Docker image for the ChatUI
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt
COPY start.sh /app
COPY api /app/api
CMD ["/bin/bash", "start.sh"]
