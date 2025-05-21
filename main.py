import sys
import yaml
from src.core.pipeline import Pipeline
from src.models import HFLLM, GoogleLLM
from src.tasks import MLMTask, DocumentRetrievalTask
from src.core import AutoTask
from dotenv import load_dotenv

load_dotenv()

def main():
    import sys
    import yaml
    from src.core.pipeline import Pipeline
    from src.core import AutoTask
    from dotenv import load_dotenv

    load_dotenv()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build and run the pipeline using AutoTask and Pipeline class
    model_provider = config["model"]["provider"]
    model_name = config["model"]["model_name"]
    model_str = f"{model_provider}:{model_name}"
    pipeline = Pipeline.build(
        f"{config['task']['type']}:{model_provider}:{model_name}",
        config=config
    )
    pipeline.run(output_cfg=config["output"])

if __name__ == "__main__":
    main()
