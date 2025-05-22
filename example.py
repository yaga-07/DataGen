import logging
from src.core.pipeline import Pipeline
from src.core import AutoModel, AutoTask
from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

available_models = AutoModel.available_models()
logger.info(f"Available models: {available_models}")

available_tasks = AutoTask.available_tasks()
logger.info(f"Available tasks: {available_tasks}")

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