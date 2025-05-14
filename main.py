import logging
from src.tasks.mlm_task import MLMTask
from src.utils.data_saver import save_data
from src.core.base_llm import BaseLLM
from src.models import HFLLM

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" #"meta-llama/Llama-3.2-3B-Instruct"

# Dummy LLM for demonstration; replace with your actual LLM implementation
class DummyLLM(BaseLLM):
    def generate_response(self, prompt):
        import json
        # Always returns a JSON list of sentences
        return json.dumps([
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming many industries."
        ])

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Initializing model and MLM task...")
    model = HFLLM(model_name=LLM_MODEL)
    logging.info(f"Using model: {print(model)}")

    domain = "technology"
    num_records = 5

    task = MLMTask(model=model, domain=domain, num_records=num_records)

    logging.info("Generating MLM data...")
    data = task.generate_data()
    logging.info(f"Generated {len(data)} records.")

    output_folder = "output"
    output_filename = "mlm_data.jsonl"
    logging.info(f"Saving data to {output_folder}/{output_filename}...")
    save_data(data, output_folder, output_filename, format="jsonl")
    logging.info("Data saved successfully.")

if __name__ == "__main__":
    main()
