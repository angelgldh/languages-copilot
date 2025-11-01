import os
import json
import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# import from the rest of the package
from scripts.utils import load_dialogues, load_config
from .constants import (
    ROLE,
    STUDENT,
    TUTOR,
    CONTENT,
    TURNS,
    INSTRUCTION,
    INPUT,
    OUTPUT,
    MESSAGE,
    TRAIN_FILE,
    TEST_FILE,
    SYSTEM_TEMPLATE,
    INSTRUCTION_TEMPLATE,
    DIALOGUE_TEMPLATE,
    CHAT_TEMPLATE,
    HISTORY_FORMAT,
    COMMON_CORRECTIONS,
    INPUT_FORMAT,
    OUTPUT_FORMAT,
    DATA,
    TRAIN_TEST_SPLIT,
    SEED,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InstructionTuningPreprocessor:
    
    def _process_turns_in_dialogue(
        self,
        turns: List[Dict[str, Any]],
        prompt_templates: Dict[str, Any],
        instruction_template: str,
    ) -> List[Dict[str, str]]:
        """
        Process a single dialogue's turns to create training examples

        Args:
            turns: List of turns in the dialogue
            prompt_templates: prompt_templates for formatting
            instruction_template: Template for instruction

        Returns:
            List of formatted examples from this dialogue
        """
        examples = []
        history = []  # Initialize history once per dialogue

        # Every turn follows the same structure: student speaks, tutor responds
        for i in range(0, len(turns) - 1, 2):
            # 1. Input and validation: check the student and tutors are well allocated 
            if i + 1 >= len(turns):  
                continue

            student_turn = turns[i]
            tutor_turn = turns[i + 1]
            if student_turn[ROLE] != STUDENT or tutor_turn[ROLE] != TUTOR:
                continue

            # 2. Format the input: student + history
            history_text = "\n".join(history)
            if history_text:
                history_text += "\n"

            input_text = prompt_templates[DIALOGUE_TEMPLATE][INPUT_FORMAT].format(
                history=history_text, student_message=student_turn[CONTENT]
            )

            # 3. Format the output: tutor message
            output_text = prompt_templates[DIALOGUE_TEMPLATE][OUTPUT_FORMAT].format(
                tutor_message=tutor_turn[CONTENT]
            )

            # 4. Format the final example and ouput
            example = {
                INSTRUCTION: instruction_template,
                INPUT: input_text,
                OUTPUT: output_text,
            }

            examples.append(example)

            # 5. Update history with the current exchange for the next iteration
            history.append(
                prompt_templates[HISTORY_FORMAT][STUDENT].format(
                    message=student_turn[CONTENT]
                )
            )
            history.append(
                prompt_templates[HISTORY_FORMAT][TUTOR].format(
                    message=tutor_turn[CONTENT]
                )
            )

        return examples

    def _format_for_instruction_tuning(
        self, dialogues: List[Dict[str, Any]], prompt_templates: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Convert dialogues to instruction tuning format

        Args:
            dialogues: List of dialogue objects
            prompt_templates: prompt_templates for formatting dialogues

        Returns:
            List of formatted examples for instruction tuning
        """
        formatted_data = []
        instruction_template = prompt_templates[INSTRUCTION_TEMPLATE]

        for dialogue in dialogues:
            # Process each dialogue to extract training examples
            dialogue_examples = self._process_turns_in_dialogue(
                dialogue[TURNS], prompt_templates, instruction_template
            )
            formatted_data.extend(dialogue_examples)

        logger.info(f"Created {len(formatted_data)} training examples")
        return formatted_data

    def _create_train_test_split(
        self, data: List[Dict[str, str]], test_ratio: float = 0.1, seed: int = 42
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split data into training and test sets

        Args:
            data: List of formatted examples
            test_ratio: Ratio of test set size to total dataset size
            seed: Random seed for reproducibility

        Returns:
            Tuple containing (training_data, test_data)
        """
        random.seed(seed)
        random.shuffle(data)

        test_size = max(1, int(len(data) * test_ratio))
        test_data = data[:test_size]
        train_data = data[test_size:]

        logger.info(
            f"Split data into {len(train_data)} training and {len(test_data)} test examples"
        )
        return train_data, test_data

    def _save_processed_data(
        self, data: List[Dict[str, str]], output_path: str
    ) -> None:
        """
        Save formatted data to a JSONL file

        Args:
            data: List of formatted examples
            output_path: Path to save the data
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(data)} examples to {output_path}")

    def preprocess_language_data(
        self,
        input_file: str,
        output_dir: str,
        prompt_templates_path: str,
        training_config_path: str,
    ) -> None:
        """
        Main preprocessing function

        Args:
            input_file: Path to raw dialogues JSONL file
            output_dir: Directory to save processed data
            prompt_templates_path: Path to prompt_templates JSON file
            training_config_path: Path to training configuration JSON file
        """
        # 1. Create output directory (if it does not exist)
        os.makedirs(output_dir, exist_ok=True)

        # 2. Load the data: all related to configs and dialogues
        prompt_templates = load_config(prompt_templates_path)
        training_config = load_config(training_config_path)
        dialogues = load_dialogues(input_file)

        # 3. Format the raw data for instruction tuning
        formatted_data = self._format_for_instruction_tuning(dialogues, prompt_templates)

        # 4. Create train/test split
        train_data, test_data = self._create_train_test_split(
            formatted_data,
            test_ratio=training_config[DATA][TRAIN_TEST_SPLIT],
            seed=training_config[DATA][SEED],
        )

        # 5. Save processed data
        train_output_path = os.path.join(output_dir, TRAIN_FILE)
        test_output_path = os.path.join(output_dir, TEST_FILE)

        self._save_processed_data(train_data, train_output_path)
        self._save_processed_data(test_data, test_output_path)

        logger.info(f"Preprocessing complete. Files saved to {output_dir}")
