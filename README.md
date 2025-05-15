# DataGen

DataGen is a minimal yet extensible framework for generating synthetic datasets using Large Language Models (LLMs). It supports both open-source and proprietary LLM providers, enabling the creation of high-quality, domain-specific data for various machine learning tasks.

## Features

- **Synthetic Data Generation:** Easily generate realistic data for NLP tasks using LLMs.
- **Pluggable LLM Providers:** Supports open-source (e.g., HuggingFace) and proprietary (e.g., Google Gemini) LLMs.
- **Task-based Design:** Each data generation scenario is encapsulated as a "task" for modularity and extensibility.
- **Easy Extension:** Add new tasks or integrate new LLM providers with minimal code changes.
- **Flexible Output:** Save generated data in JSONL, CSV, or Parquet formats.

## Project Structure

```
DataGen/
├── main.py                  # Entry point for data generation
├── src/
│   ├── core/                # Abstract base classes for LLMs and tasks
│   ├── models/              # LLM provider implementations (HuggingFace, Google, etc.)
│   ├── prompts/             # Prompt templates for different tasks
│   ├── tasks/               # Task implementations (e.g., MLM)
│   └── utils/               # Utilities (logging, data saving, etc.)
├── output/                  # Generated data output (gitignored)
├── .env                     # Environment variables for API keys and credentials
└── README.md                # Project documentation
```

## How It Works

1. **Select an LLM Provider:** Choose from supported LLMs (e.g., HuggingFace, Google Gemini).
2. **Choose a Task:** Define the data generation task (currently, Masked Language Modeling).
3. **Configure Domain & Size:** Specify the domain (e.g., "technology") and number of records.
4. **Run the Pipeline:** The framework generates data using the selected LLM and task.
5. **Save the Output:** Data is saved in your chosen format (JSONL, CSV, or Parquet).

## Example Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys and credentials.

3. **Run the main script:**
   ```bash
   python main.py
   ```

   By default, this will generate 100 MLM records in the "technology" domain using the configured LLM provider.

4. **Output:**
   - The generated data will be saved in the `output/` directory as `mlm_data.jsonl`.

## Extending DataGen

### Adding a New Task

1. Create a new task class in `src/tasks/` inheriting from `BaseTask`.
2. Implement the `generate_data()` method.
3. Add prompt templates in `src/prompts/` if needed.
4. Update `main.py` to use your new task.

### Adding a New LLM Provider

1. Create a new model class in `src/models/` inheriting from `BaseLLM`.
2. Implement the `generate_response()` method.
3. Register your model in `src/models/__init__.py`.

## Supported LLM Providers

- **HuggingFace Inference API:** Use open-source models hosted on HuggingFace.
- **Google Gemini (Vertex AI):** Use Google's proprietary LLMs via API key or service account.

## Supported Tasks

- **Masked Language Modeling (MLM):** Generates sentences with masked tokens for pretraining or evaluation.

_More tasks can be added in the future!_

## Configuration

All sensitive credentials and configuration are managed via the `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service_account.json
GOOGLE_PROJECT_ID=your_project_id
GOOGLE_LOCATION=your_location
```

## Logging

DataGen uses a colorized logger for better readability during development and debugging.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for new tasks, LLM providers, or improvements.

---

**Note:** This project is intended for research and educational purposes. Please ensure compliance with the terms of service of any LLM provider you use.
