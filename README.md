# Aspect-Based Sentiment Analysis of Airline Tweets Using BERT and BERTopic for Fine-Grained Feedback

This repository focuses on **Aspect-Based Sentiment Analysis (ABSA)** for airline tweets, aiming to extract more granular insights from customer feedback. It leverages state-of-the-art transformer models (BERT, RoBERTa, DistilBERT) combined with data preprocessing and augmentation techniques. While the project title suggests `BERTopic` integration, the current implementation primarily focuses on fine-grained sentiment classification using transformer models and rule-based aspect extraction.

It provides implementations of:
- A comprehensive **preprocessing pipeline** including text cleaning, lemmatization, stop-word removal, and data augmentation.
- **Aspect extraction** using spaCy to identify key nouns and their modifiers from tweets.
- Custom PyTorch `Dataset` for efficient handling of text, aspects, and labels, formatted for transformer models.
- **Fine-tuning** popular pre-trained transformer models (BERT, RoBERTa, DistilBERT) for a 3-class (negative, neutral, positive) sentiment classification task.
- **K-Fold cross-validation** for robust model training and evaluation.
- **Visualization tools** to compare model performance, including confusion matrices, accuracy comparisons, and training/validation loss curves.

**Author:** Nazifa Tasnim Shifa (@NazifaTasnimShifa)
**License:** MIT

---

## 📑 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration & Tips](#configuration--tips)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project provides a complete workflow for performing Aspect-Based Sentiment Analysis (ABSA) on airline tweets. The goal is to classify the sentiment (negative, neutral, positive) of a tweet with respect to specific aspects mentioned. This fine-grained approach offers deeper insights into customer feedback compared to general sentiment analysis, allowing for targeted improvements based on specific service or product features.

The pipeline involves several key stages:

-   **Data Preprocessing and Augmentation:** Raw tweet texts are thoroughly cleaned (emojis, contractions, URLs, user mentions, hashtags removed), lemmatized, and common stop-words are eliminated. Data augmentation using synonym replacement is applied to balance class distribution, especially for under-represented sentiments (positive and neutral in the provided dataset).
-   **Aspect Extraction:** Utilizing spaCy's linguistic capabilities, the system identifies potential aspects within the cleaned tweets by extracting noun phrases that are modified by adjectives. If no specific aspects are found, a generic "other" aspect is assigned.
-   **Custom Dataset Implementation:** A PyTorch `ABSADataset` is created to efficiently prepare the data for transformer models. It combines the extracted aspect with the cleaned tweet text, formatted as "aspect [SEP] text", and tokenizes it appropriately.
-   **Transformer Model Fine-tuning:** Popular pre-trained transformer models (BERT, RoBERTa, DistilBERT) are fine-tuned for a 3-class sentiment classification task. K-Fold cross-validation is employed to ensure the robustness and generalization of the model evaluations.
-   **Performance Evaluation and Visualization:** Models are evaluated using standard classification metrics (accuracy, F1-score). The results are visualized through various charts, including confusion matrices for each model, an overall accuracy comparison chart, cross-validated accuracy box plots, and training/validation loss curves.

### Core Scripts
-   `preprocess.py` — Handles data loading, text cleaning, data augmentation, and rule-based aspect extraction.
-   `dataset.py` — Defines the `ABSADataset` for preparing data samples for PyTorch `DataLoader`s.
-   `train.py` — Implements the K-Fold cross-validation training loop, model saving, and loading logic.
-   `main.py` — Orchestrates the entire project workflow: loads processed data, initializes and trains/loads transformer models, and triggers the visualization of results.
-   `visualize.py` — Generates and saves various comparison charts and metrics, including confusion matrices and loss curves.

---

## Results

After executing the `main.py` script, the project generates various visualizations and metrics comparing the performance of the fine-tuned transformer models. These outputs are saved in the `outputs/` directory.

Key results include:
-   **Confusion Matrices:** For each fine-tuned model (BERT, RoBERTa, DistilBERT), illustrating the distribution of true vs. predicted sentiment labels (negative, neutral, positive).
-   **Accuracy Comparison:** A bar chart comparing the overall accuracy of BERT, RoBERTa, DistilBERT, and a hardcoded "Proposed Model" reference point (with an accuracy of 0.91).
-   **Cross-Validated Model Comparison:** A box plot showing the distribution of accuracies across the K-folds for each model, providing insight into model stability.
-   **Training and Validation Loss Curves:** For each model, depicting how the training and validation loss evolves over the training epochs, helping to monitor for overfitting.
-   **Model Metrics CSV:** A file (`outputs/model_metrics.csv`) containing a summary of model accuracies.

**Example Visualizations (Generated in `outputs/`):**
*(Note: No static image examples are provided in this README, as these plots are dynamically generated upon script execution.)*
-   `cm_bert.png`
-   `accuracy_comparison.png`
-   `loss_curves_roberta.png`
-   `model_metrics.csv`

---

## Quickstart

### Tested With
-   Python **3.9+** (e.g., Python 3.9, 3.10, 3.11)
-   PyTorch **2.6.0**
-   HuggingFace Transformers **4.51.3**
-   NLTK **3.9.1**, spaCy **3.7.5**

### Installation
```bash
# (Recommended) Create a new virtual environment
# conda create -n absa-airline-tweets python=3.10 -y && conda activate absa-airline-tweets
# or
# python -m venv absa-env && source absa-env/bin/activate # For Linux/macOS
# python -m venv absa-env && .\absa-env\Scripts\activate # For Windows

# Install required Python packages
pip install -r requirements.txt

# Additionally, download spaCy's English model and NLTK's stopwords data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

**`requirements.txt` content:**
```txt
pandas==2.2.3
numpy==1.26.4
regex==2024.11.6
emoji==2.14.0
contractions==0.1.73
nltk==3.9.1
spacy==3.7.5
nlpaug==1.1.11
scikit-learn==1.5.2
transformers==4.51.3
torch==2.6.0
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1
shap==0.46.0
```

### Dataset

The project expects a dataset named `Tweets.csv` to be present in the root directory of the repository. A sample `Tweets.csv` file is included, which will be used by `preprocess.py`.

### GPU Usage

*   The scripts are configured to automatically leverage **CUDA if available** for PyTorch operations, significantly speeding up training.
*   If you have multiple GPUs and wish to specify a particular one, you can set the `CUDA_VISIBLE_DEVICES` environment variable:
    ```bash
    export CUDA_VISIBLE_DEVICES=0 # Use the first GPU
    ```

---

## Usage

Follow these steps to run the sentiment analysis pipeline from data preprocessing to model evaluation and visualization:

### 1. Preprocess the Data

First, execute the `preprocess.py` script. This crucial step cleans the raw tweets, performs data augmentation, extracts aspects, and prepares the data for model training.

```bash
python preprocess.py
```
Upon successful execution, this script will:
-   Load the `Tweets.csv` dataset.
-   Clean the `text` column, apply lemmatization, and remove stopwords.
-   Augment positive (label `2`) and neutral (label `1`) sentiment samples by 1000 new entries each to mitigate class imbalance.
-   Extract aspects from the `cleaned_text` using spaCy.
-   Save the fully processed DataFrame to `outputs/processed_data.csv`. A confirmation message "Processed data saved to outputs/processed_data.csv" will be printed.

### 2. Train and Evaluate Models

After preprocessing, run the `main.py` script. This script orchestrates the model training, evaluation using K-Fold cross-validation, and the generation of performance visualizations.

```bash
python main.py
```
This script will:
-   Load the `outputs/processed_data.csv` file.
-   Initialize and fine-tune BERT, RoBERTa, and DistilBERT models on the processed data.
-   Perform 3-fold cross-validation for each model.
-   If `force_retrain` is `False` (default), it will attempt to load previously saved models from `outputs/models/`. Otherwise, it will train new models.
-   Save the trained models and their tokenizers for each fold in respective subdirectories under `outputs/models/` (e.g., `outputs/models/BERT/fold_0/`).
-   Generate various performance metrics and visualizations (confusion matrices, accuracy comparisons, loss curves) and save them in the `outputs/` directory.
-   A final message "Processing complete. Visualizations saved in outputs/" will be printed.

**Retraining Models:**
By default, `main.py` is configured to load already trained models if they exist. To force the retraining of all models from scratch (e.g., after making code changes or to test different hyperparameters), modify the `force_retrain` flag in `main.py` to `True`:
```python
# main.py
# ...
force_retrain = True  # Set to True to force retraining of models
# ...
```

---

## Project Structure

```plaintext
.
├── dataset.py             # Defines the ABSADataset class for PyTorch DataLoaders.
├── main.py                # Main execution script; orchestrates preprocessing, training, and visualization.
├── preprocess.py          # Script for data cleaning, augmentation, and rule-based aspect extraction.
├── train.py               # Implements the K-Fold cross-validation training and evaluation logic for transformer models.
├── visualize.py           # Handles the generation and saving of various comparison charts and metrics.
├── Tweets.csv             # The raw input dataset containing airline tweets.
├── requirements.txt       # Lists all Python package dependencies.
├── README.md              # This project documentation file.
└── outputs/               # Directory for all generated files (created automatically).
    ├── processed_data.csv # The cleaned, augmented, and aspect-extracted dataset.
    ├── models/            # Contains saved fine-tuned models and tokenizers for each fold.
    │   ├── BERT/
    │   │   ├── fold_0/
    │   │   ├── fold_1/
    │   │   └── fold_2/
    │   ├── RoBERTa/
    │   │   └── ...
    │   └── DistilBERT/
    │       └── ...
    ├── cm_bert.png        # Confusion matrix for BERT.
    ├── accuracy_comparison.png # Bar chart comparing model accuracies.
    ├── cv_scores_comparison.png # Box plot of cross-validation accuracies.
    ├── loss_curves_bert.png    # Training and validation loss curve for BERT.
    ├── loss_curves_roberta.png # Training and validation loss curve for RoBERTa.
    ├── loss_curves_distilbert.png # Training and validation loss curve for DistilBERT.
    └── model_metrics.csv  # CSV file summarizing model performance metrics.
```

---

## Configuration & Tips

*   **Model Training Parameters (`train.py`):**
    *   `epochs`: The number of training epochs for each fold (default `3`).
    *   `lr`: The learning rate for the AdamW optimizer (default `2e-5`).
    *   `folds`: The number of folds to use for K-Fold cross-validation (default `3`).
    *   `max_length`: The maximum sequence length for tokenization, defined in `dataset.py` (default `128`). Adjust this based on the typical length of your tweets.
*   **Data Augmentation Settings (`preprocess.py`):**
    *   `aug_p`: The probability for synonym replacement in `nlpaug.augmenter.word.SynonymAug` (default `0.3`).
    *   The number of augmented samples created for positive and neutral classes is set to `1000` each. These values can be adjusted to further balance the dataset or to experiment with different augmentation strategies.
*   **Output Management:** All generated intermediate files (like `processed_data.csv`) and final outputs (trained models, plots, metrics CSVs) are saved into the `outputs/` directory. This directory is automatically created if it does not exist at the start of the script execution.
*   **Forcing Retraining (`main.py`):** The `force_retrain` flag in `main.py` is a convenient toggle. Set it to `True` if you've made changes to the model architecture, hyperparameters, or preprocessing steps and want to ensure a fresh training run. Otherwise, setting it to `False` allows for quick loading of pre-trained models.
*   **Aspect Extraction Customization:** The current aspect extraction method in `preprocess.py` is a rule-based approach (identifying `NOUN (ADJ)` pairs). For more sophisticated or domain-specific aspect identification, this function can be modified or replaced with more advanced NLP techniques (e.g., dependency parsing-based extraction or supervised aspect extraction models).

---

## Dependencies

The project relies on a set of essential Python libraries for data processing, natural language understanding, deep learning, and visualization. You can install all required packages using the provided `requirements.txt` file.

*   Python **3.9** or higher (recommended for compatibility with recent library versions)
*   `pandas`: For efficient data manipulation and analysis.
*   `numpy`: For numerical operations, especially with arrays and matrices.
*   `regex`, `emoji`, `contractions`: Essential for robust text cleaning and normalization.
*   `nltk`: Used for its comprehensive list of English stopwords.
*   `spacy`: A powerful library for advanced natural language processing, crucial for lemmatization and rule-based aspect extraction (`en_core_web_sm` model is required).
*   `nlpaug`: For advanced text data augmentation techniques, specifically synonym replacement.
*   `scikit-learn`: Provides utilities for K-Fold cross-validation, and classification report generation.
*   `transformers`: The core library for loading and fine-tuning pre-trained BERT, RoBERTa, and DistilBERT models.
*   `torch`: The foundational deep learning framework used for model training and inference.
*   `matplotlib`, `seaborn`, `plotly`: Used for generating high-quality static and interactive data visualizations.
*   `shap`: (Included in `requirements.txt`) A package for explaining machine learning model outputs, though not explicitly used in the provided code snippets, it suggests potential for future interpretability analysis.

To install:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

---

## Acknowledgements

This project benefits significantly from the open-source community and the following powerful libraries and resources:

-   **Hugging Face Transformers:** For providing an incredibly accessible and robust platform for working with state-of-the-art pre-trained language models like BERT, RoBERTa, and DistilBERT.
-   **PyTorch:** The flexible deep learning framework that underpins the model training and inference processes in this project.
-   **spaCy:** For its efficient and comprehensive natural language processing capabilities, which were instrumental in text lemmatization and the rule-based aspect extraction.
-   **NLTK:** For its readily available linguistic resources, specifically the English stopword lists.
-   **nlpaug:** For enabling effective text data augmentation to improve model generalization and handle class imbalance.
-   **scikit-learn:** For its robust tools for machine learning, including data splitting for cross-validation and performance metric calculation.
-   **Matplotlib, Seaborn, and Plotly:** For their extensive functionalities in creating insightful data visualizations and comparison charts.
-   **Dataset:** The `Tweets.csv` dataset, which forms the basis for this aspect-based sentiment analysis study.

---

## Citation

If you use this code or concepts from this repository in your research or project, please cite:

```bibtex
@software{Shifa_Hasan_Khan_ABSA_Airline_Tweets_2025,
  author  = {Nazifa Tasnim Shifa and S. M. Mehedi Hasan and Md. Iqbal Haider Khan},
  title   = {Aspect-Based Sentiment Analysis of Airline Tweets Using BERT and BERTopic for Fine-Grained Feedback},
  year    = {2025},
  url     = {https://github.com/NazifaTasnimShifa/BERT-BERTopic-ABSA},
  note    = {MIT License}
}
```

---

## License

MIT License © Nazifa Tasnim Shifa (@NazifaTasnimShifa)

See the [LICENSE](LICENSE) file for full details.