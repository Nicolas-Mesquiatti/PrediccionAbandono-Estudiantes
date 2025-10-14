# Predicción de Abandono Estudiantil - Instituto Tecnológico Beltrán

Este proyecto implementa un modelo de árbol de decisión para predecir el abandono académico en estudiantes terciarios, utilizando datos del primer año de cursada. Fue desarrollado como parte del trabajo práctico grupal para la materia **Aprendizaje Automático** .

---

##  Objetivo

Identificar estudiantes con riesgo de abandono para que la institución pueda aplicar estrategias de acompañamiento académico y vocacional.

---

###  Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `arbol_optimizado.png` | Visualización gráfica del árbol de decisión entrenado con los parámetros óptimos |
| `Predicciones_Optimizadas.xlsx` | Archivo Excel con las predicciones de abandono para los estudiantes actuales |

###  Archivos de Entrada Requeridos

| Archivo | Descripción |
|---------|-------------|
| `TablaPrediccionAbandono-Entrenamiento.xlsx` | Dataset histórico para entrenar el modelo |
| `TablaPrediccionAbandono-DatosFinal.xlsx` | Dataset de estudiantes actuales para predecir |

---



---

## ⚙️ Descripción del código

El script `Abandono.py` realiza las siguientes etapas:

1. **Carga y exploración de datos**  
   - Codificación de la variable objetivo (`EstadoFinal`)
   - Estadísticas generales del conjunto

2. **Preprocesamiento**  
   - Codificación de variables categóricas con `LabelEncoder`
   - División en conjunto de entrenamiento y prueba (80/20)

3. **Entrenamiento del modelo**  
   - Árbol de decisión con criterio `entropy`
   - Parámetros ajustados manualmente para evitar sobreajuste

4. **Evaluación del modelo**  
   - Precisión: **55%**
   - Reporte de clasificación y matriz de confusión

5. **Visualización**  
   - Generación de imagen del árbol (`arbol_optimizado.png`)

6. **Predicción sobre nuevos datos**  
   - Aplicación del modelo a estudiantes actuales
   - Exportación de resultados a `Predicciones_Optimizadas.xlsx`

7. **Ejemplo de uso**  
   - Simulación de un estudiante ficticio
   - Predicción de riesgo y recomendación institucional

---

## Resultados

- **Precisión del modelo**: 55%
- **Recall clase "Continúa"**: 89%
- **Recall clase "Abandona"**: 27%
- El modelo tiende a clasificar como "Continúa", lo que reduce falsas alarmas pero omite algunos casos de abandono.

---

##  Recomendaciones institucionales

- Priorizar tutorías para estudiantes con promedio < 6
- Acompañar a jóvenes con pocas materias aprobadas
- Monitorear carreras con alta tasa de abandono
- Usar el modelo como herramienta de apoyo, no como diagnóstico definitivo

---

##  Requisitos

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl

Instalación rápida:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```
##  Autores

**Equipo de Desarrollo:**
- Nicolás Mesquiatti
- Lucas Oviedo
- Marco Medina
- Coral Tolazzi
- Ariel Colatto
- Cristian Monzón

**Docente:** Yanina Scudero  
**Institución:** Instituto Tecnológico Beltrán
