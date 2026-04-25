FROM python:3.11-slim

ENV APP_VERSION=1.0.0
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install server dependencies
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy all application files
COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
