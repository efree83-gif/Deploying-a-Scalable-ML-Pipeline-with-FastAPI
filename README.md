# Deploying a Scalable ML Pipeline with FastAPI

A production-ready machine learning pipeline and RESTful API built with Python, Scikit-learn, and FastAPI, complete with automated CI/CD linting and testing via GitHub Actions.

## Project Overview
This project trains a classification model on census data to predict whether income exceeds $50K/yr. The workflow includes:
- Data cleaning and preprocessing pipelines.
- Model training and serialization (`model.pkl`, `encoder.pkl`).
- Sliced model performance evaluation.
- A RESTful FastAPI application for real-time model inference.
- Automated CI/CD pipeline enforcing `flake8` linting and `pytest` execution on every push.

---

## Repository Structure
```text
├── .github/workflows/main.yml   # GitHub Actions CI workflow
├── data/                        # Dataset directory (census.csv)
├── ml/                          # Core ML code (data processing, training, model)
├── model/                       # Serialized model and encoder artifacts
├── main.py                      # FastAPI application entry point
├── test_ml.py                   # Unit tests for ML functions and FastAPI endpoints
├── train_model.py               # Model training script
├── local_api.py                 # Script for local API integration testing
└── requirements.txt             # Python package dependencies
