from typing import Dict, List, Tuple, Any, Optional
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dialogues(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dialogues from a JSONL file

    Args:
        file_path (str): Path to the JSONL file containing dialogues

    Returns:
        List[Dict[str, Any]]: List of dialogue objects
    """
    dialogues = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dialogue = json.loads(line)
                dialogues.append(dialogue)

    logger.info(f"Loaded {len(dialogues)} dialogues from {file_path}")
    return dialogues
