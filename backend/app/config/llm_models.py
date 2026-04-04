# This file contains the available LLM models for each provider and their configurations. It serves
# as a central place to manage the models that can be used in the application, allowing for easy
# retrieval of model information based on the selected provider.

# The structure of the `llm_models` dictionary is as follows:
# {
#     "ProviderName": [
#         {
#             "name": "ModelName",
#             "think": True/False,  # Optional configuration for the model
#             ... # Other model-specific configurations can be added here
#         },
#         ...
#     ],
#     ...
# }

llm_models = {
    "Ollama": [
        {
            "name": "llama3.2:latest",
            "think": False
        }
    ]
}
