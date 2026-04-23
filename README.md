# Codenames LLM Bias

## Project Structure

```
codenames-llm-bias/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py                   # FastAPI route definitions
│   │   ├── core/
│   │   │   ├── llm/
│   │   │   │   ├── client.py               # LLM API client (abstract base)
│   │   │   │   └── client_local.py         # Local LLM client (Ollama)
│   │   │   ├── engine.py                   # Game engine logic
│   │   │   ├── llm_service.py              # LLM interaction service
│   │   │   └── loader.py                   # Data loader utilities
│   │   ├── models/
│   │   │   ├── game_schemas.py             # Game Pydantic models
│   │   │   ├── llm_errors.py               # LLM error types
│   │   │   └── llm_schemas.py              # LLM request/response Pydantic models
│   │   ├── templates/                      # HTML templates
│   │   │   ├── partials/
│   │   │   │   ├── _card.html              # Game card component
│   │   │   │   ├── _clue_banner.html       # Clue display banner
│   │   │   │   ├── _game_stats.html        # Game statistics panel
│   │   │   │   └── _log_entry.html         # Game log entry template
│   │   │   ├── base.html                   # Base HTML layout
│   │   │   ├── footer.html                 # Footer HTML layout
│   │   │   ├── game_base.html              # Game page base layout
│   │   │   ├── game_config.html            # Game configuration form
│   │   │   ├── game_header.html            # In-game header
│   │   │   ├── game.html                   # Main game view
│   │   │   ├── index.html                  # Landing page
│   │   │   └── main_header.html            # Main site header
│   │   ├── config.py                       # Available language models for each provider
│   │   └── main.py                         # FastAPI app entrypoint
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── boards/                             # Game boards
│   │   └── example_board.json
│   └── prompt_templates/                   # LLM prompt templates
│       ├── SYSTEM_TEMPLATE_CLUE_GIVER.txt
│       ├── SYSTEM_TEMPLATE_GUESSER.txt
│       ├── USER_TEMPLATE_CLUE_GIVER.txt
│       └── USER_TEMPLATE_GUESSER.txt
├── tests/                                  # Engine and LLMService tests
│   └── unit/
│       ├── conftest.py
│       ├── test_engine.py
│       ├── test_game_schemas.py
│       ├── test_llm_client_local.py
│       ├── test_llm_schemas.py
│       ├── test_llm_service.py
│       └── test_loader.py
├── .devcontainer/
│   └── devcontainer.json
├── .env.example                            # .env example file
├── .gitignore
├── docker-compose.yml
├── pyproject.toml                          # Python project definition
└── README.md                               # This file. Hi there! 👋
```
