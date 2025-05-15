import logging
from src.tasks.mlm_task import MLMTask
from src.utils.data_saver import save_data
from src.core.base_llm import BaseLLM
from src.models import HFLLM
from src.utils.color_logger import get_color_logger

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" #"meta-llama/Llama-3.2-3B-Instruct"

def main():
    logger = get_color_logger(level=logging.DEBUG)

    # Set mlm_task logger level to match main logger level
    mlm_logger = logging.getLogger("mlm_task")
    mlm_logger.setLevel(logger.level)

    logger.info("Initializing model and MLM task...")
    model = HFLLM(model_name=LLM_MODEL)
    logger.info(f"Using model: {model}")

    domain = "technology"
    num_records = 100

    task = MLMTask(model=model, domain=domain, num_records=num_records)

    logger.info("Generating MLM data...")
    data = task.generate_data()
    logger.info(f"Generated {len(data)} records.")

    output_folder = "output"
    output_filename = "mlm_data.jsonl"
    logger.info(f"Saving data to {output_folder}/{output_filename}...")
    save_data(data, output_folder, output_filename, format="jsonl")
    logger.info("Data saved successfully.")

if __name__ == "__main__":
    main()
