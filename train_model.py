import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Cargar y limpiar datos
df = pd.read_csv('salarydataset.csv')
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    'phD': 'PhD'
})
df = df.dropna(subset=['Salary'])

X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Modelos (bonus: comparación)
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LinearRegression())])

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# Entrenar y evaluar
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

print("Linear Regression:")
print(f"MAE: {mean_absolute_error(y_test, lr_pred):.2f}")
print(f"R²: {r2_score(y_test, lr_pred):.4f}")

print("\nRandom Forest:")
print(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}")
print(f"R²: {r2_score(y_test, rf_pred):.4f}")

# Guardar el mejor modelo (Random Forest)
joblib.dump(rf_pipeline, 'model/salary_model.pkl')
print("\nModelo Random Forest guardado en model/salary_model.pkl")
