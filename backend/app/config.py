# This file contains the available LLM models for each provider and their configurations. It serves
# as a central place to manage the models that can be used in the application, allowing for easy
# retrieval of model information based on the selected provider.
# It is also used by ollama_entrypoint.sh to automatically pull the specified models when the Ollama
# container starts.

# The structure of the `llm_models` dictionary is as follows:
# {
#     "ProviderName": {
#         "ModelName": {
#             "think": True/False,  # Optional configuration for the model
#             ... # Other model-specific configurations can be added here
#         },
#         ...
#     },
#     ...
# }

llm_models = {
    "Ollama": {
        # LLaMA
        "llama3.1:latest": {
            "think": False
        },
        # Mistral
        "mistral-small3.2:latest": {
            "think": False
        },
    },
    "OpenRouter": {
        "x-ai/grok-4.3": {},
        "google/gemini-3.5-flash": {},
    },
}

# The exact local (Ollama) weights this batch is validated against, keyed by the model tag as it
# appears in `llm_models["Ollama"]` (and in a seat's `model_name`). These are the full manifest
# digests captured from the running daemon's /api/tags; the daemon rejects a digest-as-model tag,
# so the tag stays `:latest` and this map is the reproducibility anchor. When digest enforcement is
# on (the batch path), a served digest that differs from the expected one below ABORTS the run
# before any game is dispatched (see game_runner._enforce_local_digests / ModelDigestMismatchError).
# Human-readable: llama3.1 = 8B Q4_K_M; mistral-small3.2 = 24B Q4_K_M.
EXPECTED_LOCAL_DIGESTS = {
    "llama3.1:latest":
        "sha256:46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e",
    "mistral-small3.2:latest":
        "sha256:5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b",
}
