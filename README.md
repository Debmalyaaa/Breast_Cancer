# 🧬 Breast Cancer Prediction using Machine Learning

## Overview

This project focuses on the prediction of breast cancer malignancy using supervised machine learning models. The objective is to build a reliable classification model that can assist in distinguishing between benign and malignant tumors based on various medical features, thereby supporting early diagnosis and clinical decision-making.

---

## Technologies Used

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, scikit-learn  
- **Algorithms:** Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN)  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (`data.csv`)

---

## Project Structure

1. **Data Ingestion & Cleaning**
   - Loaded dataset and removed irrelevant features.
   - Checked for and handled missing values.

2. **Exploratory Data Analysis**
   - Analyzed distribution of features across diagnosis classes.
   - Visualized feature correlations to identify strong predictors.

3. **Data Preprocessing**
   - Standardized numerical features using `StandardScaler`.
   - Encoded categorical target variable.

4. **Model Development**
   - Trained multiple classification models.
   - Evaluated model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

5. **Model Comparison**
   - Compared performance across classifiers.
   - Selected the best-performing model based on both predictive accuracy and generalization.

---

## Key Observations

- **Class Balance:** The dataset is fairly balanced between benign and malignant cases.
- **Top Features:** Attributes such as `mean radius`, `concave points`, and `texture` show high correlation with the diagnosis outcome.
- **Model Performance:** Random Forest and SVM demonstrated the highest accuracy, offering robust performance on test data.

---

## Results Summary

| Model                | Accuracy   |
|---------------------|------------|
| Logistic Regression | xx.xx%     |
| Decision Tree       | xx.xx%     |
| Random Forest       | xx.xx% ✅  |
| Support Vector SVM  | xx.xx% ✅  |
| K-Nearest Neighbors | xx.xx%     |

✅ Indicates top-performing models based on accuracy and reliability.

---

## Running the Project

1. Open `breast cancer.ipynb` using **Google Colab** or **Jupyter Notebook**.
2. Ensure `data.csv` is placed in the working directory.
3. Execute the notebook sequentially to perform data processing, model training, and evaluation.

**Google Colab Users:**  
Use the following snippet to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/path_to_your_dataset/data.csv')
