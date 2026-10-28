#!/bin/bash

# Start the Ollama server in the background
ollama serve &

# Wait for the Ollama server to be ready
echo "Waiting for Ollama server to start..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done

# The models to pull: the keys of llm_models["Ollama"] in config.py (mounted as /llm_models.py).
# This image has no Python, so awk reads them: skipping comment lines, it follows the braces from
# the "Ollama" entry and prints each key one level inside it. Matching any quoted "name:tag" string
# instead would also pick up the sha256 digests of EXPECTED_LOCAL_DIGESTS.
MODELS=$(awk '
    /^[ \t]*#/ { next }
    !in_ollama && /"Ollama"[ \t]*:/ { in_ollama = 1 }
    in_ollama {
        if (depth == 1 && match($0, /"[^"]+"[ \t]*:/)) {
            key = substr($0, RSTART + 1, RLENGTH - 1)
            sub(/"[ \t]*:$/, "", key)
            print key
        }
        depth += gsub(/[{]/, "{") - gsub(/[}]/, "}")
        if (depth == 0) exit
    }' /llm_models.py)

if [ -z "$MODELS" ]; then
    echo "No Ollama models found in /llm_models.py. Nothing to pull."
fi

for MODEL_NAME in $MODELS; do
    # Exact match on the NAME column: a substring match would take llama3.1:8b-instruct-q8_0 for
    # llama3.1:8b.
    if ! ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "$MODEL_NAME"; then
        echo "Model $MODEL_NAME not found. Pulling..."
        ollama pull "$MODEL_NAME"
    else
        echo "Model $MODEL_NAME already exists. Skipping pull."
    fi
done

echo "All specified models are ready."

# Keep the container alive by waiting on the background ollama server
wait