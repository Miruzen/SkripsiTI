# LongFormer Classification Model

## Model Overview
- **Model Type**: LongFormer
- **Task**: Text Classification
- **Framework**: Hugging Face Transformers
- **Selected Checkpoint**: checkpoint-714

## Model Details
- **Base Model**: allenai/longformer-base-4096
- **Max Sequence Length**: 4096 tokens
- **Model Size**: ~149M parameters
- **Training Data**: LF_Labelled.csv

## Training Information
- **Training Date**: October 17-18, 2023
- **Hardware Used**: GPU (CUDA)
- **Training Steps**: 714
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5

## Model Files
- `model.safetensors`: Model weights
- `config.json`: Model architecture configuration
- `tokenizer.json`: Tokenizer configuration
- `vocab.json`: Vocabulary file
- `merges.txt`: BPE merges
- `optimizer.pt`: Optimizer state
- `trainer_state.json`: Training state and metrics

## Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "best_model/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

## Citation
```bibtex
@misc{longformer_classification_2023,
  author = {[Miruzen]},
  title = {LongFormer Text Classification Model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

Note: Replace the placeholder information in brackets with your specific details.