# 🎯 Algoritmos de Clasificación en Python

Este módulo implementa visualizaciones interactivas de los algoritmos de clasificación más populares en Machine Learning utilizando scikit-learn.

## 📋 Contenido

### 1. K-Nearest Neighbors (K-NN)
La función `plot_knn()` visualiza cómo el algoritmo K-NN clasifica puntos en un espacio bidimensional.

**Características principales:**
- Implementa clasificación con k=3 vecinos
- Genera regiones de decisión visuales
- Utiliza un mapa de calor para mostrar las áreas de clasificación
- Incluye los puntos de datos originales coloreados por clase

### 2. Regresión Logística
La función `plot_logistic_regression()` muestra la separación binaria de clases.

**Características principales:**
- Visualiza la curva logística de probabilidad
- Muestra el límite de decisión en 0.5
- Representa los puntos de datos de cada clase
- Incluye la probabilidad de pertenencia a cada clase

### 3. Árbol de Decisión
La función `plot_decision_tree()` ilustra la estructura jerárquica de decisiones.

**Características principales:**
- Profundidad máxima de 3 niveles
- Nodos coloreados según la clase mayoritaria
- Muestra las reglas de decisión en cada nodo
- Visualización clara de la estructura del árbol

### 4. Random Forest
La función `plot_random_forest()` visualiza un conjunto de árboles de decisión.

**Características principales:**
- Muestra 3 árboles independientes
- Profundidad máxima de 3 por árbol
- Visualización paralela de los árboles
- Permite comparar las diferentes estructuras

## 🛠️ Requisitos

```python
numpy==1.21.0
matplotlib==3.4.3
scikit-learn==0.24.2

pip install numpy matplotlib scikit-learn
```

## 🚀 Uso

```python
# Importar las funciones necesarias
from visualizations import plot_knn, plot_logistic_regression, plot_decision_tree, plot_random_forest

# Generar visualizaciones individuales
plot_knn()
plot_logistic_regression()
plot_decision_tree()
plot_random_forest()
```

## 📊 Ejemplos de Salida

Cada función generará una visualización detallada del algoritmo correspondiente:
- KNN: Muestra regiones de decisión y puntos clasificados
- Regresión Logística: Muestra la curva de probabilidad y el límite de decisión
- Árbol de Decisión: Muestra la estructura completa del árbol
- Random Forest: Muestra tres árboles diferentes del ensemble

## 📝 Notas
- Los datos son generados sintéticamente para demostración
- Las visualizaciones son interactivas cuando se ejecutan en un notebook
- Los colores y estilos pueden personalizarse según necesidades