import yaml
import logging
from src.models import HFLLM, GoogleLLM
from src.tasks.mlm_task import MLMTask
from src.utils.data_saver import save_data
from src.utils.color_logger import get_color_logger
from dotenv import load_dotenv
import os
from datetime import datetime

def run_from_config(config_path: str):
    load_dotenv()
    logger = get_color_logger(level=logging.INFO)
    logger.info(f"Loading config from {config_path}...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Model selection
    model_cfg = config["model"]
    provider = model_cfg["provider"].lower()
    model_name = model_cfg["model_name"]
    logger.info(f"Using provider: {provider}, model: {model_name}")

    if provider == "google":
        model = GoogleLLM(
            model_name=model_name,
            api_key=model_cfg.get("api_key"),
            service_account_json=model_cfg.get("service_account_json")
        )
        llm_short = "google"
    elif provider == "hf" or provider == "huggingface":
        model = HFLLM(
            model_name=model_name,
            api_key=model_cfg.get("api_key")
        )
        llm_short = "hf"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Task selection
    task_cfg = config["task"]
    task_type = task_cfg["type"].lower()
    domain = task_cfg["domain"]
    num_records = task_cfg["num_records"]

    if task_type == "mlm":
        task = MLMTask(model=model, domain=domain, num_records=num_records)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    logger.info(f"Generating data for task: {task_type}...")
    data = task.generate_data()
    logger.info(f"Generated {len(data)} records.")

    # Output
    output_cfg = config["output"]
    output_folder = output_cfg.get("folder", "output")
    output_format = output_cfg.get("format", "jsonl")
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_type}_{llm_short}_{num_records}_{dt_str}.{output_format}"
    logger.info(f"Saving data to {output_folder}/{filename} ...")
    save_data(
        data,
        folder=output_folder,
        filename=filename,
        format=output_format
    )
    logger.info(f"Data saved successfully at {os.path.join(output_folder, filename)}.")
