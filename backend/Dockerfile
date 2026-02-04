FROM python:3.11-slim

# Create a non-root user to run the application
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -l -u ${USER_ID} -g appuser -m appuser

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R appuser:appuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

COPY --chown=appuser:appuser . .

# run FastAPI with automatic reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]