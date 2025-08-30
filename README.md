❤️ Heart Disease Prediction using Logistic Regression


📌 Project Overview
This project aims to predict the risk of heart disease in the next 10 years using the Framingham Heart Study dataset from Kaggle.We use Logistic Regression as the primary model, handle missing values, perform feature selection, and apply RandomizedSearchCV for hyperparameter tuning.  
🔗 Dataset Link: Heart Disease Prediction - Kaggle

⚙️ Technologies Used

🐍 Python  
📊 Pandas & NumPy → Data handling  
📈 Matplotlib & Seaborn → Visualization  
🤖 Scikit-learn → Modeling & Hyperparameter Tuning


📂 Dataset Information



Detail
Value



Rows
4238


Columns
16


Target Column
TenYearCHD (0 = No Heart Disease, 1 = Heart Disease)


Class Distribution
0 → 3594  1 → 644 (Imbalanced)


🔎 Missing Values
Columns with missing values:education, cigsPerDay, BPMeds, totChol, BMI, heartRate, glucose  
✅ Strategy: Filled with mean/mode since we cannot afford to lose data.  

🛠️ Project Workflow
1. Data Preprocessing

Checked dataset shape → (4238, 16)  
Handled missing values with mean/mode imputation  
No duplicates found  
Defined X (features) and y (target)

2. Train-Test Split

Split dataset 80% train / 20% test

3. Baseline Logistic Regression

Model: LogisticRegression  
Accuracy: 0.833  
Confusion Matrix:  
✅ Correctly predicted many cases  
❌ 141 False Negatives → Struggles to detect positive heart disease cases



📊 (Confusion Matrix image attached in repo)  
4. Hyperparameter Tuning (RandomizedSearchCV)
param_dist = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': stats.loguniform(1e-3, 1e2),
    'solver': ['liblinear', 'saga'],
    'max_iter': [500],
    'class_weight': ['balanced', None],
    'l1_ratio': [0.1, 0.5, 0.9]  # only for elasticnet
}

5. 🔍 Results
Best Score: 0.8355Best Params:  

C = 0.0019  
penalty = elasticnet  
solver = saga  
max_iter = 500  
class_weight = None  
l1_ratio = 0.1

6. Feature Selection (SelectKBest + Chi2)

Selected Top 10 Features  

Logistic Regression accuracy: 0.8337  
RandomizedSearchCV tuned model accuracy: 0.85  
(Confusion Matrix shows accuracy is misleading due to imbalance)


Reduced to Top 7 Features  

Retrained model with 7 features  
Accuracy improved slightly  
RandomizedSearchCV best score: 0.853  
Params: C=0.09, penalty=l2, solver=saga, l1_ratio=0.9




📊 Model Performance Summary



Experiment
Features Used
Accuracy
Best Hyperparameters



Logistic Regression (Default)
All (16)
0.833
-


RandomizedSearchCV (All)
All (16)
0.8355
ElasticNet, C=0.0019


Logistic + SelectKBest
Top 10
0.8337
-


RandomizedSearchCV (Top 10)
Top 10
0.85
L2, C=0.099


Logistic + SelectKBest
Top 7
0.834
-


RandomizedSearchCV (Top 7)
Top 7
0.853
L2, C=0.09



📉 Limitations

⚠️ Dataset is imbalanced, model misses positive cases (false negatives)  
High accuracy but low recall for minority class (Heart Disease)  
Logistic Regression may not be the best for imbalanced medical data


🚀 Next Steps

Try Decision Tree, Random Forest, XGBoost  
Apply SMOTE (Oversampling) for balancing data  
Evaluate using Precision, Recall, F1-Score, ROC-AUC instead of just accuracy


📌 Conclusion

Logistic Regression gives ~85% accuracy  
Feature selection + tuning improves performance  
Still, false negatives remain a big problem → critical in healthcare predictions


🖼️ Visualizations

Confusion Matrix (Baseline)  
Confusion Matrix (Feature Selected)  
Accuracy Comparison Graphs

(All plots are included in repository)  

🤝 Contribution

Fork 🍴 this repo  
Create a new branch 🌿  
Make your changes  
Submit a Pull Request ✅


📜 License
This project is licensed under the MIT License
