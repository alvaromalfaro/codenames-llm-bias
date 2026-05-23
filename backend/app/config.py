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
        "llama3.2:latest": {
            "think": False
        },
        # Gemma4
        "gemma4:latest": {
            "think": False
        },
        # Mistral
        "magistral:latest": {
            "think": False
        },
    },
    "OpenRouter": {
        "meta-llama/llama-3.3-70b-instruct:free": {},
        "openai/gpt-oss-120b:free": {},
        "minimax/minimax-m2.5:free": {},
        "deepseek/deepseek-v4-flash:free": {},
    },
}
