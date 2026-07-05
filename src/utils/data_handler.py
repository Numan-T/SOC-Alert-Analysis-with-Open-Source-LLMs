import json
import pandas as pd
from pathlib import Path
from src.config import main_config as conf


def load_json(file_path: Path):
    """Loads a JSON file and returns it's content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, file_path: Path):
    """Saves a dictionary as JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_txt(file_path: Path) -> str:
    """Returns the content of a text file as string."""
    return file_path.read_text(encoding="utf-8").strip()


def load_alerts(file_path: Path) -> list:
    """
    Loads alerts from json or excel file and returns them in a uniform json format with 'alert_id' and 'content' fields.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.json':
        return load_json(file_path)

    elif file_path.suffix.lower() == '.xlsx':
        return _load_alerts_from_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported data format of alerts file: {file_path.suffix}")


def _load_alerts_from_excel(file_path: Path) -> list:
    """
    Parser for excel alerts file.
    """
    sheet_name = conf.TEST_ALERTS_SHEET_NAME
    try:
        df = pd.read_excel(file_path, sheet_name)
    except Exception as e:
        raise ValueError(
            f"Couldn't read sheet {sheet_name} of Excel file: {e}")

    required_columns = ['Alert_id', 'Alert']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"One of these columns is missing: {required_columns}")

    alerts = []
    for _, row in df.iterrows():
        # Skip empty rows
        if pd.isna(row['Alert_id']) or pd.isna(row['Alert']):
            continue

        alert_id = str(row['Alert_id']).strip().replace('.0', '')
        alert_content_raw = row['Alert']

        # Parse 'Alert' cell as json
        try:
            if isinstance(alert_content_raw, str) and alert_content_raw.strip().startswith('{'):
                alert_content = json.loads(alert_content_raw)
            else:
                alert_content = alert_content_raw
        except json.JSONDecodeError:
            print(
                f"WARNING: JSON parsing failed for alert {alert_id}. Using raw data instead.")
            alert_content = alert_content_raw

        # Create dictionary with 'alert_id' and 'content'
        alerts.append({
            "alert_id": alert_id,
            "content": alert_content
        })

    print(f"Successfully loaded {len(alerts)} alerts from Excel.")
    return alerts
