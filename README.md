cat > README.md << 'EOF'
# Salary Predictor Web Application

## Project Title & Description
A web application that uses machine learning to predict annual salary based on age, gender, education level, job title, and years of experience. Built with Flask and Scikit-Learn.

## Dataset Description
- Source: Provided salarydataset.csv (downloaded from public repository)
- Features used: Age, Gender, Education Level, Job Title, Years of Experience
- Target variable: Salary

## Machine Learning Approach
- Models used: Linear Regression and Random Forest Regressor (bonus: comparison)
- Chosen model: Random Forest (better performance)
- Pipeline: Handles missing values (imputer), encodes categorical variables (OneHotEncoder), scales numeric features (StandardScaler)
- Evaluation metrics (on test set):
  - Linear Regression: MAE 15386.70, R² 0.8437
  - Random Forest: MAE 3037.34, R² 0.9778

## Web Application
- Flask app with HTML form and result page
- Routes: '/' (form), '/predict' (POST for prediction)
- How prediction works: Loads pickled model, creates DataFrame from user input, calls predict()
- Bonus: Job Titles dropdown loaded dynamically from dataset

## Installation & Execution
1. pip install -r requirements.txt
2. python train_model.py
3. python app.py (run at http://127.0.0.1:5000)

## Results & Screenshots
![Form Page](screenshots/form_page.png)
![Prediction Result](screenshots/prediction_result.png)
![GitHub Repo](screenshots/github_repo.png)
![Code Structure](screenshots/code_structure.png)

## Reflection
- What feature influences salary the most? Years of Experience (highest correlation)
- What limitation does your model have? Small dataset and high cardinality in Job Title may cause overfitting
- What would you improve with more time? Hyperparameter tuning for Random Forest, add more data, or use XGBoost
EOF
