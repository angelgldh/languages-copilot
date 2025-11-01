# Languages Copilot

A conversational application for language learning.

## Raw Data Format

The project leverages first examples of student-tutor conversation produced with another language model. This data includes:

- A list of conversation turns between students and tutors
- Each turn includes a role ("student" or "tutor") and content (the message text)

Example:
```json
{
  "turns": [
    {"role": "student", "content": "¿Cómo estás hoy?"},
    {"role": "tutor", "content": "Estoy muy bien, gracias. ¿Y tú?"},
    {"role": "student", "content": "Estoy bien, pero tengo una pregunta sobre verbos."},
    {"role": "tutor", "content": "Claro, dime tu pregunta sobre los verbos."}
  ]
}
```

## Project Structure

```
languages-copilot/
├── backend/
│   ├── api/
│   │   ├── routers/
│   │   ├── core/
│   │   └── models/
│   └── tests/
├── client/
│   ├── components/
│   ├── api/
│   └── utils/
└── training/
    ├── data/              # Raw and processed dialogue data
    ├── scripts/           # Scripts for preprocessing, fine-tuning, and evaluation
    ├── configs/           # Configuration files for models, training, and prompts
    ├── notebooks/         # Jupyter notebooks for analysis and experimentation
    ├── logs/              # Training logs and metrics
    └── outputs/           # Fine-tuned model weights and artifacts
```
