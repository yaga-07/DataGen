import sys
import yaml
from src.pipeline import Pipeline
from src.models import HFLLM, GoogleLLM
from src.tasks import MLMTask, DocumentRetrievalTask
from src.core import AutoTask
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build and run the pipeline using AutoTask and Pipeline class
    model = GoogleLLM(config["model"]["model_name"]) # Example: "gemini-1.5-pro"
    task = AutoTask.get_task(
        config["task"]["type"], # e.g., "mlm" or "document_retrieval"
        model, 
        config["task"]["domain"], # e.g., "AI"
        config["task"]["num_records"] # e.g., 10
        )
    
    pipeline = Pipeline(task=task)
    pipeline.run(output_cfg=config["output"])
