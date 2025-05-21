import logging
from src.core.pipeline import Pipeline
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