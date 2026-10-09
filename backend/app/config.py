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
        "llama3.1:8b": {
            "think": False
        },
        # Gemma
        "gemma4:12b": {
            "think": False
        },
        # Qwen
        "qwen2.5:14b": {
            "think": False
        },
        # Mistral
        "mistral-small3.2:24b": {
            "think": False
        },
    },
    # "OpenRouter": {
    #     "x-ai/grok-4.3": {},
    #     "google/gemini-3.5-flash": {},
    # },
}

# The exact local (Ollama) weights this batch is validated against, keyed by the model tag as it
# appears in `llm_models["Ollama"]` (and in a seat's `model_name`). These are the full manifest
# digests captured from the running daemon's /api/tags; the daemon rejects a digest-as-model tag,
# so the tag stays the PLAYED tag (the exact key in `llm_models["Ollama"]`, e.g. `llama3.1:8b` -
# never a bare `:latest`, which would not match what is played) and this map is the reproducibility
# anchor. When digest enforcement is on (the batch path), a served digest that differs from the
# expected one below ABORTS the run before any game is dispatched (see game_runner._enforce_local_digests /
# ModelDigestMismatchError).
# Human-readable: llama3.1 = 8B Q4_K_M; mistral-small3.2 = 24B Q4_K_M.
EXPECTED_LOCAL_DIGESTS = {
    "llama3.1:8b":
        "sha256:46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e",
    "gemma4:12b":
        "sha256:4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
    "qwen2.5:14b":
        "sha256:7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6",
    "mistral-small3.2:24b":
        "sha256:5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b",
}

# The context window, in tokens, of every local (Ollama) request. Set here so that every machine
# plays with the same one: left unset, ollama picks it from the GPU's memory (4096 on a 16 GB card),
# so the same batch would run with a different context on another machine. The longest exchange of
# the test games (prompt plus answer) took 1925 tokens.
OLLAMA_NUM_CTX = 8192

# The most tokens a model may write in one answer, for every provider (ollama's num_predict,
# OpenRouter's max_tokens). The longest answer of the test games took 576 tokens. A model that
# loops (repeating "Let's try X. No.") is cut here, and the cut answer is drawn again (see
# LLMTruncatedResponseError) instead of failing the game on a broken JSON.
LLM_MAX_OUTPUT_TOKENS = 2048
