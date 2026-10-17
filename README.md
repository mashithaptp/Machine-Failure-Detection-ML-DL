# Machine Failure Detection Using Machine Learning and Deep Learning

## Project Overview

This project aims to predict industrial machine failures using Machine Learning and Deep Learning techniques. The objective is to identify potential machine failures from operational sensor measurements and support predictive maintenance decisions.

The project uses the AI4I 2020 Predictive Maintenance Dataset and compares a Gradient Boosting machine learning classifier with an Artificial Neural Network (ANN).

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository  
**Total Records:** 10,000  
**Target Variable:** Machine failure

The input features include:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Temperature difference (engineered feature)
- Mechanical power proxy (engineered feature)

Failure-mode indicators are excluded from the prediction inputs to avoid target leakage.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib and Seaborn
- Streamlit
- Jupyter Notebook

## Models Implemented

**Machine Learning:** Gradient Boosting Classifier

**Deep Learning:** Artificial Neural Network (ANN)

The models are evaluated using a held-out test dataset, with particular attention to detecting the minority failure class.

## Model Performance

| Metric | Gradient Boosting | ANN |
|---|---:|---:|
| Accuracy | 99.00% | 97.87% |
| Precision | 83.33% | 65.08% |
| Recall | 88.24% | 80.39% |
| F1 Score | 85.71% | 71.93% |
| ROC-AUC | 0.9754 | 0.9811 |
| PR-AUC | 0.9207 | 0.7980 |

Gradient Boosting achieved better precision, recall, F1 score, and PR-AUC on the held-out test set. The ANN achieved a slightly higher ROC-AUC.

Gradient Boosting was selected for deployment in the Streamlit application.

## Streamlit Application

The project includes a Streamlit web application for predicting machine failure from user-provided machine operating measurements.

### How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Project Files

- `ai4i2020.csv` — Dataset
- `Machine_Failure_Detection_ML_and_Deep_Learning.ipynb` — Model development and evaluation
- `machine_failure_detection_model.joblib` — Saved Gradient Boosting prediction model and configuration
- `app.py` — Streamlit application
- `requirements.txt` — Application dependencies

## Conclusion

This project demonstrates the use of Machine Learning and Deep Learning for industrial machine failure detection. Based on the evaluated metrics, Gradient Boosting provided the strongest overall failure-class detection performance among the two compared approaches.

The predictions are intended as a demonstration using the AI4I 2020 synthetic dataset and would require further validation before real-world industrial deployment.
