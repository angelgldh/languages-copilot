import argparse
from scripts.preprocessing.instruction_tuning_preprocessor import InstructionTuningPreprocessor
from scripts.preprocessing.constants import TRAIN_FILE, VALIDATION_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dialogue data for language tutor training"
    )

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to raw dialogues JSONL file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/data/processed",
        help="Directory to save processed data",
    )

    parser.add_argument(
        "--templates_path",
        type=str,
        default="training/configs/prompt_templates.json",
        help="Path to templates JSON file",
    )

    parser.add_argument(
        "--training_config_path",
        type=str,
        default="training/configs/training_config.json",
        help="Path to training configuration JSON file",
    )

    args = parser.parse_args()

    # Create preprocessor instance and preprocess the data
    preprocessor = InstructionTuningPreprocessor()
    preprocessor.preprocess_language_data(
        args.input_file, args.output_dir, args.templates_path, args.training_config_path
    )
