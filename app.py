from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Cargar modelo y datos
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

# Ruta para limpiar gráficas antiguas
PLOT_PATH = 'static/plots/prediction_plot.png'

@app.route('/')
def home():
    # Limpiar gráfica anterior si existe
    if os.path.exists(PLOT_PATH):
        os.remove(PLOT_PATH)
    return render_template('index.html', job_titles=job_titles, genders=genders, educations=educations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        age = float(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        job_title = request.form['job_title']
        years_exp = float(request.form['years_exp'])

        # Crear DataFrame para predicción
        data = {
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [years_exp]
        }
        df_input = pd.DataFrame(data)
        prediction = model.predict(df_input)[0]

        # === Generar gráfica de regresión lineal ===
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Years of Experience'], df['Salary'], alpha=0.6, color='blue', label='Datos reales')

        # Línea de regresión simple (solo para visualización)
        import numpy as np
        z = np.polyfit(df['Years of Experience'], df['Salary'], 1)
        p = np.poly1d(z)
        plt.plot(df['Years of Experience'], p(df['Years of Experience']), color='red', linewidth=2, label='Línea de tendencia')

        # Marcar el punto del usuario
        plt.scatter(years_exp, prediction, color='green', s=200, label=f'Tu predicción: ${prediction:,.0f}', zorder=5)
        plt.title('Relación entre Años de Experiencia y Salario', fontsize=16)
        plt.xlabel('Años de Experiencia', fontsize=12)
        plt.ylabel('Salario Anual ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Guardar gráfica
        plt.savefig(PLOT_PATH)
        plt.close()

        return render_template('result.html',
                               prediction=round(prediction, 2),
                               age=age,
                               gender=gender,
                               education=education,
                               job_title=job_title,
                               years_exp=years_exp,
                               plot_exists=True)
    except Exception as e:
        return render_template('index.html',
                               error="Por favor verifica los datos ingresados",
                               job_titles=job_titles, genders=genders, educations=educations)

if __name__ == '__main__':
    app.run(debug=True)
