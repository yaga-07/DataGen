import json
import os
from typing import Any, List, Dict, Union
import pandas as pd

def save_data(
    data: Union[List[Dict], List[List], Dict, List[Any]],
    folder: str,
    filename: str,
    format: str = "jsonl"
) -> None:
    """
    Save data to disk in the specified format and folder.

    Args:
        data: The data to save. Can be a list of dicts, list of lists, dict, or list of any serializable objects.
        folder: The output folder path.
        filename: The output file name.
        format: One of 'jsonl', 'csv', or 'parquet'.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    if format == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, list):
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    elif format == "csv":
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame([data])
        df.to_csv(path, index=False)
    elif format == "parquet":
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame([data])
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
