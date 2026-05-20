# Deep Learning Course Projects

This repository contains two deep learning course projects:

1. **Image Classification with Stanford Cars**
2. **Turkish News Classification with BERTurk and Decision Agent**

The projects were developed as part of a deep learning course. The first project focuses on image classification and CNN architecture modifications. The second project focuses on Turkish natural language processing, transformer fine-tuning, and a rule-based decision agent.

---

## Repository Structure
text
deep_learning_projects/
│
├── README.md
├── .gitignore
│
├── image-classification/
│   ├── README.md
│   ├── src/
│   └── outputs/
│
└── nlp-news-classification/
    ├── README.md
    ├── src/
    └── outputs/

Project 1: Image Classification

The image classification project uses the Stanford Cars dataset. The goal is to classify vehicle images and compare different CNN-based approaches.

The project includes:

Baseline CNN / AlexNet-like model
Custom CNN architectures named CarNet
CarNet v1, v2, and v3 experiments
Data augmentation experiments
ResNet18 transfer learning
Full 196-class Stanford Cars experiment
Evaluation with Accuracy, Macro-F1, Classification Report, Confusion Matrix, and Top-5 Accuracy

Project folder:

image-classification/
Project 2: Turkish News Classification and Decision Agent

The NLP project focuses on Turkish news classification. The goal is to classify Turkish news articles into 10 categories and then use the model output inside a decision agent.

The project includes:

Dataset preparation
TF-IDF + Logistic Regression baseline
BERTurk fine-tuning
Tokenization examples
Classification report and confusion matrix
Detailed error analysis
Reliability diagram and calibration analysis
Decision agent based on confidence score and Top-1 / Top-2 margin

Project folder:

nlp-news-classification/
Important Notes

Large files are not included in this repository.

The following files and folders are excluded:

.venv/
data/
outputs/models/
trained model weights
raw datasets

The repository includes:

source code
training scripts
model architecture code
evaluation reports
selected output figures
CSV/JSON result files
README reports

Pretrained models such as ResNet18 and BERTurk are downloaded automatically by the related libraries when the scripts are executed.

Requirements

The main Python packages used in these projects are:

torch
torchvision
transformers
datasets
pandas
numpy
scikit-learn
matplotlib
tqdm
pillow
joblib
requests

A virtual environment is recommended.

Example setup:

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Summary

These two projects demonstrate different applications of deep learning:

In the image classification project, CNN architectures were modified and compared with transfer learning.
In the NLP project, classical machine learning and transformer-based models were compared, and a decision agent was added to make routing decisions based on model confidence.

The details of each project are explained in their own README files.
