# Trainer
A tiny framework for deeplearning training and validating

## Code Structure
- template
    - dataset: dataset implementation
    - train_utils
        - networks: networks structures goes here
        - optim: optimizers
        - schedulers : optimizer steppers
        - logger.py: tensorboard logger
        - tv_scheduler.py: The scheduler for training and validation
    - main.py
- code_generate.py: generate project in a path

## Usage

```bash
python code_generate.py --code_path ./mnist
```