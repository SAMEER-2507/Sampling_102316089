# Sampling_Assignment_102316089

## Title: Credit Card Fraud Detection - Model Evaluation with Sampling Techniques

This repository evaluates the performance of multiple machine learning models on a credit card fraud detection dataset. Different sampling techniques are applied to balance the dataset and measure the impact on model performance.

---

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Sampling Techniques](#sampling-techniques)
* [Models Used](#models-used)
* [Methodology](#methodology)
* [Files & Output](#files--output)
* [Results](#results)

---

## Introduction

This project applies several sampling techniques to address class imbalance in a credit card fraud detection dataset. After balancing, five machine learning models are trained and evaluated on each sampled dataset to compare accuracy across sampling strategies.

**Sampling techniques applied**

* Simple Random Sampling
* Systematic Sampling
* Stratified Sampling
* Cluster Sampling
* Bootstrap Sampling

**Models evaluated**

* RandomForest
* HistGradientBoosting
* Support Vector Machine (SVM)
* Logistic Regression
* K-Nearest Neighbors (KNN)

The objective is to quantify how different sampling methods affect model accuracy and to provide reproducible code for comparison.

---

## Installation

Recommended environment: Python 3.8+ (Google Colab is supported).

Install required packages:

```bash
pip install -U pandas numpy scikit-learn imbalanced-learn
```

(If running on Google Colab, prefix `!` before `pip`.)

---

## Sampling Techniques

* **Simple Random Sampling**: Randomly select a subset of rows without replacement.
* **Systematic Sampling**: Select rows at fixed intervals from an ordered list (interval ≈ N / sample_size).
* **Stratified Sampling**: Preserve the class proportions by sampling within each class. Useful when class distributions must be maintained.
* **Cluster Sampling**: Partition the dataset into clusters (KMeans used here) and select whole clusters as the sample.
* **Bootstrap Sampling**: Sample with replacement to create a dataset of the target size (useful for model variance estimation).

---

## Models Used

* **RandomForestClassifier** — ensemble tree-based method, robust to feature scaling.
* **HistGradientBoostingClassifier** — gradient boosting optimized for numerical data.
* **SVC (SVM)** — kernel-based classifier (sensitive to scaling).
* **LogisticRegression** — linear classifier, good baseline.
* **KNeighborsClassifier** — distance-based method, sensitive to feature scaling and dimensionality.

All models are trained with reasonable default hyperparameters; adjust hyperparameters for production experiments.

---

## Methodology

Pipeline followed in this project:

1. **Data collection** → obtain `Creditcard_data.csv`.
2. **Data preprocessing** → encode categoricals, scale numeric features where appropriate.
3. **Data balancing** → SMOTE oversampling to produce `Creditcard_data_balanced.csv`.
4. **Sampling** → produce five sample files using different sampling techniques.
5. **Model training** → train five models on each sample (train/test split with stratification where applicable).
6. **Model testing** → evaluate models on held-out test sets.
7. **Evaluation & results** → save accuracy results for comparison.

---

## Files

* `Creditcard_data.csv` — original dataset (user-provided).
* `Creditcard_data_balanced.csv` — balanced dataset produced by oversampling.
* `sample_simple_random.csv`
* `sample_systematic.csv`
* `sample_stratified.csv`
* `sample_cluster.csv`
* `sample_bootstrap.csv` — five sampled datasets.

---

## Results

The project saves model accuracy results to results folder. Open that folder to review and compare the accuracy of each model across sampling techniques. Use these results to identify sampling-method/model combinations that perform best on the chosen dataset and preprocessing pipeline.
