cat > README.md << 'EOF'
![Banner del Proyecto - Predicción de Salario](screenshots/banner.png)

# Aplicación Web: Predicción de Salarios con Machine Learning

¡Proyecto completo y funcional! Predice salarios anuales usando datos reales con un modelo de Random Forest.

**Tecnologías usadas**: Flask • Scikit-Learn • Bootstrap • GitHub

---

## Descripción del Proyecto
Aplicación web que permite predecir el salario anual de una persona según su:
- Edad
- Género
- Nivel educativo
- Cargo laboral
- Años de experiencia

Cumple **todos los requisitos obligatorios** + **los dos bonuses**:
- Comparación de modelos (Linear Regression vs Random Forest)
- Menús desplegables de cargos cargados dinámicamente desde el dataset

![Formulario de entrada](screenshots/form_page.png)

## Dataset
- **Archivo**: `salarydataset.csv`
- **Características**: Age, Gender, Education Level, Job Title, Years of Experience
- **Objetivo**: Salary (salario anual)

## Modelo de Machine Learning
- **Pipeline completo** con:
  - Manejo de valores faltantes
  - Codificación categórica (OneHotEncoder)
  - Escalado numérico
- **Modelos comparados**:
  - Regresión Lineal → MAE ≈ 15,387 | R² ≈ 0.84
  - **Random Forest** (elegido) → MAE ≈ 3,037 | R² ≈ 0.98
- Modelo guardado en `model/salary_model.pkl`

## Aplicación Web (Flask)
- Formulario con entrada de datos
- Predicción clara y profesional
- Entradas del usuario repetidas en el resultado
- Interfaz moderna con Bootstrap

![Resultado de predicción de ejemplo](screenshots/prediction_result.png)

## Instalación y Ejecución
```bash
pip install -r requirements.txt
python train_model.py    # Entrena el modelo
python app.py            # Inicia la app en http://127.0.0.1:5000
