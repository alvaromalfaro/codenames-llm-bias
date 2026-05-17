FROM python:3.12-slim

# Create a non-root user to run the application
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -l -u ${USER_ID} -g appuser -m appuser

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R appuser:appuser /app

# Install only the runtime dependencies declared in pyproject.toml, without
# installing the backend package itself. This avoids placing a stub in
# site-packages that would shadow the volume-mounted source at runtime.
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools && python3 -c "import tomllib,subprocess,sys;deps=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies'];subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir']+deps)"

RUN python3 -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data')"

USER appuser

COPY --chown=appuser:appuser . .

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
