import sys
from src.runner import run_from_config

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_from_config(config_path)
