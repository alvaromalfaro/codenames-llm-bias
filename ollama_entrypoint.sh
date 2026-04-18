#!/bin/bash

# Start the Ollama server in the background
ollama serve &

# Wait for the Ollama server to be ready
echo "Waiting for Ollama server to start..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done

# Extract Ollama model names from llm_models.py (only "name:tag" quoted strings)
MODELS=$(grep -oP '"\K[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+(?=")' /llm_models.py)

for MODEL_NAME in $MODELS; do
    if ! ollama list | grep -q "$MODEL_NAME"; then
        echo "Model $MODEL_NAME not found. Pulling..."
        ollama pull "$MODEL_NAME"
    else
        echo "Model $MODEL_NAME already exists. Skipping pull."
    fi
done

echo "All specified models are ready."

# Keep the container alive by waiting on the background ollama server
wait