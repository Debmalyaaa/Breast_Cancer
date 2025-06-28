# Breast Cancer Diagnosis Prediction

## Objective
This project aims to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on various tumor characteristics. The model helps support early and accurate diagnosis, which is crucial for effective treatment planning.

## Tech Stack
- **Programming Language**: Python 3  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Seaborn, Matplotlib  
- **Machine Learning**: scikit-learn  
- **Feature Selection**: SelectKBest  
- **Model**: Random Forest Classifier  

## Dataset
The Wisconsin Breast Cancer Diagnostic Dataset contains:
- 569 instances (357 benign, 212 malignant)
- 30 features computed from digitized images of fine needle aspirates (FNA)
- Target variable: diagnosis (M = malignant, B = benign)

## Project Workflow
1. **Data Collection & Exploration**
   - Loaded and examined dataset structure
   - Checked for missing values (none found)
   - Analyzed basic statistics

2. **Feature Engineering**
   - Dropped unnecessary columns (id, Unnamed: 32)
   - Identified 30 numerical features
   - Encoded target variable (M=1, B=0)

3. **Data Preprocessing**
   - Split data into train/test sets (80/20 ratio)
   - Applied StandardScaler for feature normalization
   - Used SelectKBest for feature selection (top 15 features)

4. **Model Training**
   - Implemented Random Forest Classifier
   - Set parameters: n_estimators=100, bootstrap=True

5. **Evaluation**
   - Achieved accuracy score on test set
   - Performed 5-fold cross-validation

## Key Findings
- The Random Forest model achieved **96.5% accuracy** on the test set
- Cross-validation scores showed consistent performance (scores: [0.96, 0.95, 0.96, 0.93, 0.96])
- Important features included radius_mean, concave points_worst, and perimeter_worst


