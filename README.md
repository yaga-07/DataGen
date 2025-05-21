# DataGen

DataGen is a minimal yet extensible framework for generating synthetic datasets using Large Language Models (LLMs). It supports both open-source and proprietary LLM providers, enabling the creation of high-quality, domain-specific data for various machine learning tasks.

## Features

- **Synthetic Data Generation:** Easily generate realistic data for NLP tasks using LLMs.
- **Pluggable LLM Providers:** Supports open-source (e.g., HuggingFace) and proprietary (e.g., Google Gemini) LLMs.
- **Task-based Design:** Each data generation scenario is encapsulated as a "task" for modularity and extensibility.
- **Easy Extension:** Add new tasks or integrate new LLM providers with minimal code changes.
- **Flexible Output:** Save generated data in JSONL, CSV, or Parquet formats.
- **Automatic Model/Task Registration:** No need to manually import or register models/tasks—DataGen dynamically discovers them.

## Project Structure

```
DataGen/
├── main.py                  # Entry point for data generation
├── src/
│   ├── core/                # Abstract base classes for LLMs and tasks
│   ├── models/              # LLM provider implementations (HuggingFace, Google, etc.)
│   ├── prompts/             # Prompt templates for different tasks
│   ├── tasks/               # Task implementations (e.g., MLM, Document Retrieval)
│   └── utils/               # Utilities (logging, data saving, etc.)
├── output/                  # Generated data output (gitignored)
├── .env                     # Environment variables for API keys and credentials
├── config.yaml              # Main configuration file for data generation
└── README.md                # Project documentation
```

## How It Works

1. **Configure the Workflow:** Edit `config.yaml` to specify the LLM provider, model, task, domain, number of records, and output format.
2. **Run the Pipeline:** The framework generates data using the selected LLM and task.
3. **Save the Output:** Data is saved in your chosen format (JSONL, CSV, or Parquet) with an auto-generated filename.

## Example Configuration (`config.yaml`)

```yaml
model:
  provider: google         # or "hf" for HuggingFace
  model_name: gemini-1.5-pro
  # api_key: your_google_api_key
  # service_account_json: /path/to/service_account.json

task:
  type: mlm
  domain: technology
  num_records: 200

output:
  folder: output
  format: jsonl
```

- **provider:** `"google"` for Google Gemini or `"hf"` for HuggingFace.
- **model_name:** Name of the LLM model.
- **task.type:** Task type, e.g., `"mlm"` for Masked Language Modeling.
- **task.domain:** Domain for sentence generation.
- **task.num_records:** Number of records to generate.
- **output.folder:** Output directory.
- **output.format:** Output file format (`jsonl`, `csv`, or `parquet`).

## Example Usage

You can use DataGen via the main entry point. Here is an example usage as found in `main.py`:

```python
import logging
from src.pipeline import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()

config = {
    "model": {
        "provider": "google",
        "model_name": "gemini-1.5-pro"
    },
    "task": {
        "type": "mlm",
        "domain": "AI",
        "num_records": 2
    },
    "output": {
        "folder": "output",
        "format": "jsonl"
    }
}

pipeline = Pipeline.build("mlm:google:gemini-1.5-pro", config=config)
pipeline.run(output_cfg=config["output"])
```

Or, if you want to use a YAML config file:

```bash
python main.py path/to/your_config.yaml
```

## Extending DataGen

### Adding a New Task

1. Create a new task class in `src/tasks/` inheriting from `BaseTask`.
2. Implement the `generate_data()` method.
3. Add prompt templates in `src/prompts/` if needed.
4. Decorate your class with `@AutoTask.register("your_task_name")`.
5. **No need to manually import or register your task—DataGen will discover it automatically.**

### Adding a New LLM Provider

1. Create a new model class in `src/models/` inheriting from `BaseLLM`.
2. Implement the `generate_response()` method.
3. Decorate your class with `@AutoLLM.register("your_provider_name")`.
4. **No need to manually import or register your model—DataGen will discover it automatically.**

## Supported LLM Providers

- **HuggingFace Inference API:** Use open-source models hosted on HuggingFace.
- **Google Gemini (Vertex AI):** Use Google's proprietary LLMs via API key or service account.

## Supported Tasks

- **Masked Language Modeling (MLM):** Generates sentences with masked tokens for pretraining or evaluation.
- **Document Retrieval:** Generates query-document pairs for retrieval-augmented generation.

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
