# Machine Learning-Based Organic Farming Suitability Prediction

BCA 5th Semester PBL Project

Predicts whether a crop–field combination is **Suitable** or **Not Suitable** for organic farming using Logistic Regression, Decision Tree, KNN and Random Forest.

## Files
- `organic_farming_ml.py` – full pipeline (EDA, preprocessing, models, metrics, feature importance)
- `organic_farming_suitability_1000.csv` – dataset
- `outputs/` – figures, result tables and the saved best model
- `Organic_Farming_ML_Project_Report.docx` – project report

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
python organic_farming_ml.py organic_farming_suitability_1000.csv
```

## Result
Best model: Logistic Regression (accuracy 80.5%, F1 0.764, ROC-AUC 0.889).

## Important
The dataset is **source-calibrated synthetic data** for educational use, not real field records. Labels come from an instructional rubric, not an official organic certification.
