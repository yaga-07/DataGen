from src.core.base_task import AutoTask
from src.models import HFLLM, GoogleLLM
from src.utils.data_saver import save_data
from src.utils.color_logger import get_color_logger
from datetime import datetime
import os

class Pipeline:
    def __init__(self, task=None):
        self.model = None
        self.task = task
        self.config = None
        self.logger = get_color_logger(level="INFO")

    def build(self, model, task, config=None):
        """
        Build the pipeline from model and task instances.
        Optionally store config for output/run.
        """
        self.model = model
        self.task = task
        self.config = config
        self.logger.info(f"Pipeline built with task '{type(task).__name__}' and model '{type(model).__name__}'.")


    def run(self, output_cfg=None):
        """
        Run the pipeline: generate data and save it.
        Uses output_cfg if provided, else falls back to self.config['output'].
        """
        if not self.task:
            raise RuntimeError("Pipeline not built. Call build(model, task, config) first.")

        if output_cfg is None:
            if self.config and "output" in self.config:
                output_cfg = self.config["output"]
            else:
                raise ValueError("No output config provided.")

        # Try to infer task/model/num_records for filename
        task_type = getattr(self.task, "task_name", type(self.task).__name__.lower())
        llm_short = type(self.task.model).__name__.replace("LLM", "").lower()
        num_records = getattr(self.task, "num_records", "N")

        self.logger.info(f"Generating data for task: {task_type}...")
        data = self.task.generate_data()
        self.logger.info(f"Generated {len(data)} records.")

        output_folder = output_cfg.get("folder", "output")
        output_format = output_cfg.get("format", "jsonl")
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_type}_{llm_short}_{num_records}_{dt_str}.{output_format}"
        self.logger.info(f"Saving data to {output_folder}/{filename} ...")
        save_data(
            data,
            folder=output_folder,
            filename=filename,
            format=output_format
        )
        self.logger.info(f"Data saved successfully at {os.path.join(output_folder, filename)}.")
