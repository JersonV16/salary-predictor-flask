from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y datos para dropdowns (bonus)
model = joblib.load('model/salary_model.pkl')
df = pd.read_csv('salarydataset.csv')
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    'phD': 'PhD'
})

job_titles = sorted(df['Job Title'].dropna().unique().tolist())
genders = sorted(df['Gender'].dropna().unique().tolist())
educations = sorted(df['Education Level'].dropna().unique().tolist())

@app.route('/')
def home():
    return render_template('index.html', job_titles=job_titles, genders=genders, educations=educations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Age': [float(request.form['age'])],
            'Gender': [request.form['gender']],
            'Education Level': [request.form['education']],
            'Job Title': [request.form['job_title']],
            'Years of Experience': [float(request.form['years_exp'])]
        }
        df_input = pd.DataFrame(data)
        prediction = model.predict(df_input)[0]

        return render_template('result.html',
                               prediction=round(prediction, 2),
                               age=data['Age'][0],
                               gender=data['Gender'][0],
                               education=data['Education Level'][0],
                               job_title=data['Job Title'][0],
                               years_exp=data['Years of Experience'][0])
    except:
        return render_template('index.html',
                               error="Por favor verifica los datos ingresados",
                               job_titles=job_titles, genders=genders, educations=educations)

if __name__ == '__main__':
    app.run(debug=True)
