This is a project for the Illuin technical challenge focused on classifying algorithmic exercises. The goal is to predict tags (e.g., "math", "graphs", "strings") for problems from a dataset based on Codeforces exercises.

### Installation

Dependencies are listed in the `pyproject.toml` file. You can install them using `uv` or `pip`.

With `uv`:
```bash
uv pip install .
```

With `pip`:
```bash
pip install .
```

### Project Structure

*   **`data/`**: This directory is used for storing datasets. Raw data should be placed in `data/code_classification_dataset/`. The preprocessing notebook generates parquet files (`dataset.parquet`, `preprocessed_dataset.parquet`) in this directory. I put the models I trained
*   in this directory.
*   **`models/`**: Trained models and preprocessing objects (TF-IDF vectorizer, scaler) are saved here as `.pkl` files by the training script.

### Notebooks

The repository contains two main notebooks that document the workflow:

1.  **`eda_and_preprocessing.ipynb`**: This notebook covers the initial exploratory data analysis (EDA) of the dataset. It includes visualizations of tag distributions, feature analysis, and the data cleaning and preprocessing steps.
2.  **`training_notebook.ipynb`**: This notebook details the model training and evaluation process. It explores several models, starting with text-only features and then building a multimodal model that incorporates problem description, source code, and difficulty level.

### Command-Line Interface Usage

The `predict_cli.py` script provides a command-line interface for training, predicting, and evaluating models. The json fomat that is expected is the same as
of the json data format provided in the challenge. 

**1. Train a model:**

You can train a text-only model (faster) or a multimodal model (more accurate). However, to train, the data must be in parquet format and preprocessed (see notebook). 

```bash
# Train a text-only model
python predict_cli.py train --train-data data/preprocessed_dataset.parquet --output models/text_model.pkl --type text

# Train a multimodal model (requires PyTorch and Transformers)
python predict_cli.py train --train-data data/preprocessed_dataset.parquet --output models/multimodal_model.pkl --type multimodal
```

**2. Make predictions:**

Use a trained model to predict tags for a single JSON file or a directory of files.

```bash
# Predict on a single sample
python predict_cli.py predict --input data/code_classification_dataset/sample_0.json --model models/text_model.pkl

# Predict on a directory of samples and save results
python predict_cli.py predict --input data/code_classification_dataset/ --model models/multimodal_model.pkl --output predictions.json
```

**3. Evaluate a model:**

Evaluate a trained model's performance on a test set that includes ground-truth labels. Similarly to predict, this also works for a single JSON file or
a directory of files.

```bash
# Evaluate the text-only model (one sample)
python predict_cli.py evaluate --input data/sample_0.json --model models/text_model.pkl

# Evaluate the text-only model (one directory)
python predict_cli.py evaluate --input data/test_samples/ --model models/text_model.pkl

```
