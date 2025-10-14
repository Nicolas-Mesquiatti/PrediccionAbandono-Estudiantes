import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. ANÁLISIS EXPLORATORIO DE DATOS

print("PREDICCIÓN DE ABANDONO ESTUDIANTIL - INSTITUTO TECNOLÓGICO BELTRÁN")

# Cargar datos
df = pd.read_excel("D:/Tecnicas De Procesamiento del habla/Materia2-Machine learning/tp arbol/Arbol-Grupal/TablaPrediccionAbandono-Entrenamiento.xlsx")

# Codificar variable objetivo
df['EstadoFinal'] = df['EstadoFinal'].map({'Continúa': 0, 'Abandonó': 1})

print(f"INFORMACIÓN GENERAL:")
print(f"   • Total de estudiantes: {len(df)}")
print(f"   • Estudiantes que abandonan: {sum(df['EstadoFinal'])} ({sum(df['EstadoFinal'])/len(df)*100:.1f}%)")
print(f"   • Estudiantes que continúan: {len(df) - sum(df['EstadoFinal'])} ({(len(df) - sum(df['EstadoFinal']))/len(df)*100:.1f}%)")

# 2. PREPROCESAMIENTO DE DATOS
# Codificar variables categóricas
label_encoders = {}
categorical_columns = ['genero', 'carrera', 'trabaja/NoTrabaja', 'ActividadesExtracurriculares(Estudio)']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Preparar características y variable objetivo
X = df.drop('EstadoFinal', axis=1)
y = df['EstadoFinal']

# División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   • Conjunto de entrenamiento: {len(X_train)} estudiantes")
print(f"   • Conjunto de prueba: {len(X_test)} estudiantes")

# 3. OPTIMIZACIÓN DEL ÁRBOL CON BUSQUEDA DE HIPERPARÁMETROS

print(f"\n3. OPTIMIZACIÓN DEL MODELO:")

# Probar diferentes combinaciones de hiperparámetros
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

arbol = DecisionTreeClassifier(random_state=42)

#Buscar mejores parámetros (comentado porque toma tiempo, pero es lo recomendado)
#grid_search = GridSearchCV(arbol, param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

#print(f"   • Mejores parámetros: {grid_search.best_params_}")
#print(f"   • Mejor precisión en validación: {grid_search.best_score_:.4f}")

# Usar parámetros optimizados manualmente basados en análisis
arbol_optimizado = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,           # Reducir profundidad para evitar sobreajuste
    min_samples_split=10,   # Más muestras para dividir
    min_samples_leaf=5,     # Más muestras en hojas
    random_state=42
)

arbol_optimizado.fit(X_train, y_train)

print("  Árbol optimizado entrenado")
print(f"   • Profundidad: {arbol_optimizado.get_depth()}")
print(f"   • Hojas: {arbol_optimizado.get_n_leaves()}")

# 4. EVALUACIÓN DEL MODELO OPTIMIZADO

print(f"\n4. EVALUACIÓN DEL MODELO OPTIMIZADO:")

# Predecir en conjunto de prueba
y_pred_opt = arbol_optimizado.predict(X_test)

# Métricas de evaluación
precision_opt = accuracy_score(y_test, y_pred_opt)
reporte_opt = classification_report(y_test, y_pred_opt, target_names=['Continúa', 'Abandona'])

print(f"   • Reporte de clasificación:")
print(f"\n{reporte_opt}")

# Matriz de confusión
cm_opt = confusion_matrix(y_test, y_pred_opt)
print(f"   • Matriz de Confusión:")
print(f"     Verdaderos Positivos: {cm_opt[1,1]} (Abandonan - correcto)")
print(f"     Falsos Positivos:    {cm_opt[0,1]} (Continúan - predicho abandono)")
print(f"     Verdaderos Negativos: {cm_opt[0,0]} (Continúan - correcto)") 
print(f"     Falsos Negativos:    {cm_opt[1,0]} (Abandonan - predicho continuación)")

# 5. ANÁLISIS DE LÍMITE DE PRECISIÓN
# Precisión del modelo "siempre predecir la clase mayoritaria"
precision_baseline = max(y_test.mean(), 1 - y_test.mean())
# Calcular potencial teórico
mejor_precision_teorica = min(0.85, precision_baseline + 0.3)  # Máximo realista 85%

# 6. VISUALIZACIÓN DEL ÁRBOL OPTIMIZADO

print(f"\n6. VISUALIZACIÓN DEL ÁRBOL OPTIMIZADO:")

nombres_legibles = {
    'edad': 'Edad',
    'genero': 'Género', 
    'carrera': 'Carrera',
    'PromedioPrimerCuatrimestre': 'Promedio 1er Cuatrimestre',
    'CantMateriasAprobadasPrimerCuatrimestre': 'Materias Aprobadas',
    'CantMateriasDesaprobadasPrimerCuatrimestre': 'Materias Desaprobadas',
    'AsistenciaPromedio(%)': 'Asistencia Promedio (%)',
    'trabaja/NoTrabaja': 'Trabaja',
    'DistanciaDomicilioAlInstituto(Kms)': 'Distancia al Instituto (Km)',
    'ActividadesExtracurriculares(Estudio)': 'Actividades Extracurriculares'
}

feature_names_legibles = [nombres_legibles.get(col, col) for col in X.columns]

plt.figure(figsize=(16, 10))
plot_tree(arbol_optimizado, 
          feature_names=feature_names_legibles,
          class_names=['Continúa', 'Abandona'],
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=True)

plt.title(f"Árbol Optimizado - Precisión: {precision_opt:.4f} ({precision_opt*100:.1f}%)", 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('arbol_optimizado.png', dpi=300, bbox_inches='tight')
plt.show()

print("   Árbol optimizado guardado")

# 7. PREDICCIÓN CON MODELO OPTIMIZADO

print(f"\n7. PREDICCIÓN CON MODELO OPTIMIZADO:")

try:
    df_nuevo = pd.read_excel("D:/Tecnicas De Procesamiento del habla/Materia2-Machine learning/tp arbol/Arbol-Grupal/TablaPrediccionAbandono-DatosFinal.xlsx")
    
    for col in categorical_columns:
        if col in df_nuevo.columns:
            df_nuevo[col] = label_encoders[col].transform(df_nuevo[col].astype(str))
    
    df_nuevo = df_nuevo[X.columns]
    
    predicciones_opt = arbol_optimizado.predict(df_nuevo)
    probabilidades_opt = arbol_optimizado.predict_proba(df_nuevo)[:, 1]
    # Convertir valores booleanos a enteros (opcional pero útil)
    df_nuevo['trabaja/NoTrabaja'] = df_nuevo['trabaja/NoTrabaja'].astype(int)

    df_resultados_opt = df_nuevo.copy()
    df_resultados_opt['Prediccion'] = predicciones_opt
    df_resultados_opt['Probabilidad_Abandono'] = probabilidades_opt
    df_resultados_opt['Estado_Predicho'] = df_resultados_opt['Prediccion'].map({0: 'Continúa', 1: 'Abandona'})
    
    print(f"   RESUMEN DE PREDICCIONES OPTIMIZADAS:")
    print(f"   • Total de estudiantes evaluados: {len(df_resultados_opt)}")
    print(f"   • Predicen Abandono: {sum(predicciones_opt)}")
    print(f"   • Predicen Continúan: {len(predicciones_opt) - sum(predicciones_opt)}")
    print(f"   • Precisión esperada: {precision_opt*100:.1f}%")
    
    df_resultados_opt.to_excel("D:/Tecnicas De Procesamiento del habla/Materia2-Machine learning/tp arbol/Arbol-Grupal/Predicciones_Optimizadas.xlsx", index=False)
    print("   Predicciones optimizadas guardadas")
    
except Exception as e:
    print(f"   Error en predicción: {e}")

print(f"ANÁLISIS COMPLETADO - Precisión: {precision_opt*100:.1f}%")

# 8. EJEMPLO DE USO - PREDICCIÓN INDIVIDUAL (VERSIÓN REFINADA)

print(f"\n8. EJEMPLO DE USO - PREDICCIÓN INDIVIDUAL:")

# Mostrar clases disponibles para cada variable categórica
print(" Clases disponibles para variables categóricas:")
for col in categorical_columns:
    print(f" • {col}: {list(label_encoders[col].classes_)}")

# Crear un estudiante ficticio con valores válidos
valores_ejemplo = {
    'genero': 'm',  # debe coincidir con las clases originales
    'carrera': 'TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL',
    'trabaja/NoTrabaja': 'Sí',
    'ActividadesExtracurriculares(Estudio)': 'No'
}

# Validar y transformar valores categóricos
estudiante_ejemplo = {
    'edad': 22,
    'PromedioPrimerCuatrimestre': 5.8,
    'CantMateriasAprobadasPrimerCuatrimestre': 3,
    'CantMateriasDesaprobadasPrimerCuatrimestre': 2,
    'AsistenciaPromedio(%)': 65,
    'DistanciaDomicilioAlInstituto(Kms)': 10
}

for col in categorical_columns:
    try:
        estudiante_ejemplo[col] = label_encoders[col].transform([valores_ejemplo[col]])[0]
    except ValueError:
        estudiante_ejemplo[col] = 0  # Valor por defecto si no se reconoce

# Convertir a DataFrame
df_ejemplo = pd.DataFrame([estudiante_ejemplo])
df_ejemplo = df_ejemplo[X.columns]  # Asegurar orden

# Predecir
prediccion = arbol_optimizado.predict(df_ejemplo)[0]
probabilidad = arbol_optimizado.predict_proba(df_ejemplo)[0][1]

# Interpretar resultado
estado = 'Abandona' if prediccion == 1 else 'Continúa'
riesgo = 'ALTO' if probabilidad > 0.7 else 'MODERADO' if probabilidad > 0.3 else 'BAJO'
accion = 'Intervención urgente' if riesgo == 'ALTO' else 'Seguimiento académico' if riesgo == 'MODERADO' else 'Monitoreo estándar'

# Mostrar resultados
print("\n🎓 ESTUDIANTE FICTICIO EVALUADO:")
print(f"{'Variable':<40} {'Valor'}")
print(f"{'Edad':<40} {estudiante_ejemplo['edad']}")
print(f"{'Carrera':<40} {valores_ejemplo['carrera']}")
print(f"{'Promedio 1er Cuatrimestre':<40} {estudiante_ejemplo['PromedioPrimerCuatrimestre']}")
print(f"{'Materias Aprobadas':<40} {estudiante_ejemplo['CantMateriasAprobadasPrimerCuatrimestre']}")
print(f"{'Materias Desaprobadas':<40} {estudiante_ejemplo['CantMateriasDesaprobadasPrimerCuatrimestre']}")
print(f"{'Asistencia Promedio (%)':<40} {estudiante_ejemplo['AsistenciaPromedio(%)']}%")
print(f"{'Distancia al Instituto (Km)':<40} {estudiante_ejemplo['DistanciaDomicilioAlInstituto(Kms)']}")
print(f"{'Trabaja':<40} {valores_ejemplo['trabaja/NoTrabaja']}")
print(f"{'Actividades Extracurriculares':<40} {valores_ejemplo['ActividadesExtracurriculares(Estudio)']}")
print(f" PREDICCIÓN: {estado} (Probabilidad de abandono: {probabilidad:.2%})")
print(f" NIVEL DE RIESGO: {riesgo}")
print(f" RECOMENDACIÓN: {accion}")