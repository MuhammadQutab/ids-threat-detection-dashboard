# Geo-Aware ML Intrusion Detection Dashboard

This repository contains an MSc-level proof-of-concept dashboard for machine learning-based intrusion detection. The system uses a trained XGBoost model to classify network traffic and present results through an interactive Streamlit dashboard.

## Live Demo

https://ids-threat-detection-dashboard-gv4uxcbmbj8zbkxqjfak4m.streamlit.app

## Project Summary

The project investigates how machine learning can support intrusion detection using selected traffic classes from the CSE-CIC-IDS2018 dataset. The final prototype classifies network traffic into five categories:

- Benign
- DDoS
- DoS
- Botnet
- BruteForce

The dashboard presents model predictions with confidence scores, severity levels, visual analytics, and geo-aware alert enrichment. Geo-location is used only as contextual enrichment and not as a model training feature.

## Final Model

Several machine learning models were compared during development, including Decision Tree, Random Forest, Logistic Regression, and XGBoost. XGBoost was selected as the final model because it achieved the strongest performance on the balanced internal test split.

| Model | Accuracy | F1-score |
|---|---:|---:|
| Decision Tree | 0.9993 | 0.9993 |
| Random Forest | 0.9987 | 0.9987 |
| Logistic Regression | 0.9931 | 0.9931 |
| XGBoost | 0.9997 | 0.9997 |

## Dashboard Features

- Upload or use sample CSV traffic data
- Validate required feature columns
- Run single-record and bulk predictions
- Display prediction confidence and severity
- Show SOC-style detection summary
- Visualise prediction and severity distribution
- Provide geo-aware alert enrichment
- Download prediction results

## Repository Contents

```text
app.py
requirements.txt
final_xgboost_ids_model.pkl
final_label_encoder.pkl
final_feature_columns.pkl
sample_test_data_100.csv
.gitignore 


## Limitations 

This is an academic prototype, not a commercial intrusion detection system. The current version uses prepared CSV files rather than live packet capture. Geo-aware enrichment is simulated/contextual and should not be interpreted as confirmed attacker location. Additional attack categories such as Web Attack and Infiltration may be included in future work.

## Academic Context 

This project was developed as part of an MSc research project exploring machine learning-based intrusion detection, model comparison, dashboard deployment, and practical alert interpretation.