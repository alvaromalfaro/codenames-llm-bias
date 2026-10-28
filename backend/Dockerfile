# The same Python as .python-version, which the local venv uses.
FROM python:3.12.13-slim-bookworm

# uv installs the locked dependencies; pinned like everything the environment is built from.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

# Create a non-root user to run the application
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -l -u ${USER_ID} -g appuser -m appuser

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R appuser:appuser /app

# Install the runtime dependencies at the exact versions of uv.lock (--frozen: fail rather than
# re-resolve), without installing the backend package itself. This avoids placing a stub in
# site-packages that would shadow the volume-mounted source at runtime.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-header -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# WordNet backs the clue validator. nltk.download writes the corpora 0600-root, which the
# non-root runtime user cannot read, so widen the permissions in the same layer.
RUN python3 -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data')"     && chmod -R a+rX /usr/local/share/nltk_data

USER appuser

COPY --chown=appuser:appuser . .

# Nothing to serve: the container stays up for `docker compose exec app python scripts/...`.
CMD ["sleep", "infinity"]
