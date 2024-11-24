# Importación de bibliotecas necesarias
import numpy as np  # Para operaciones numéricas y matrices
import matplotlib.pyplot as plt  # Para crear visualizaciones
from sklearn.neighbors import KNeighborsClassifier  # Algoritmo K-NN
from sklearn.linear_model import LogisticRegression  # Algoritmos de regresión
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Árbol de decisión y su visualización
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.datasets import make_classification  # Para generar datos sintéticos

# Configuración básica de matplotlib
plt.style.use('default')  # Usar el estilo predeterminado de matplotlib

def plot_knn():
    """
    Visualiza el algoritmo K-Nearest Neighbors (K-NN) en un espacio bidimensional.
    Muestra cómo el algoritmo divide el espacio en regiones de decisión basadas en los k vecinos más cercanos.
    """
    # Generar datos sintéticos para clasificación binaria (dos clases)
    # n_samples: número de puntos, n_features: número de características
    # n_redundant: características redundantes, n_informative: características informativas
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                            n_informative=2, random_state=1, 
                            n_clusters_per_class=1)
    
    # Crear y entrenar el modelo K-NN con 3 vecinos
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)  # Ajustar el modelo a los datos
    
    # Crear una malla de puntos para visualizar las regiones de decisión
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Predecir la clase para cada punto en la malla
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Crear la visualización
    plt.figure(figsize=(10, 8))
    # Dibujar las regiones de decisión con un mapa de calor
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    # Dibujar los puntos de datos originales
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu')
    plt.colorbar(scatter)  # Agregar barra de color
    plt.title('Clasificación K-NN (k=3)')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.show()

def plot_logistic_regression():
    """
    Visualiza la regresión logística mostrando cómo separa dos clases
    y cómo calcula las probabilidades de pertenencia a cada clase.
    """
    # Generar datos sintéticos para clasificación binaria
    X, y = make_classification(n_samples=100, n_features=1, n_classes=2, 
                            n_clusters_per_class=1, flip_y=0.03, n_redundant=0,
                            n_informative=1, random_state=1)
    
    # Crear y entrenar el modelo de regresión logística
    model = LogisticRegression()
    model.fit(X, y)
    
    # Visualización
    plt.figure(figsize=(10, 8))
    # Dibujar los puntos de datos de cada clase
    plt.scatter(X[y == 0], y[y == 0], color='blue', label='Clase 0')
    plt.scatter(X[y == 1], y[y == 1], color='red', label='Clase 1')
    
    # Generar puntos para la curva de decisión
    X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred = model.predict_proba(X_test)[:, 1]  # Probabilidades de clase 1
    
    # Dibujar la curva logística y el límite de decisión
    plt.plot(X_test, y_pred, color='green', label='Curva logística')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Límite de decisión')
    plt.title('Regresión Logística')
    plt.xlabel('Característica')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_tree():
    """
    Visualiza un árbol de decisión mostrando cómo el algoritmo divide
    el espacio de características usando reglas de decisión.
    """
    # Generar datos sintéticos
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                            n_informative=2, random_state=1,
                            n_clusters_per_class=1)
    
    # Crear y entrenar el árbol de decisión con profundidad máxima de 3
    tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    tree.fit(X, y)
    
    # Visualizar el árbol
    plt.figure(figsize=(15, 10))
    plot_tree(tree, filled=True, feature_names=['X1', 'X2'],
            class_names=['Clase 0', 'Clase 1'])
    plt.title('Árbol de Decisión')
    plt.show()

def plot_random_forest():
    """
    Visualiza un Random Forest mostrando tres árboles de decisión
    que forman parte del conjunto (ensemble).
    """
    # Generar datos sintéticos
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                            n_informative=2, random_state=1,
                            n_clusters_per_class=1)
    
    # Crear y entrenar el Random Forest
    rf = RandomForestClassifier(n_estimators=3, max_depth=3)  # 3 árboles con profundidad 3
    rf.fit(X, y)
    
    # Visualizar los tres árboles
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, tree in enumerate(rf.estimators_[:3]):
        plot_tree(tree, filled=True, feature_names=['X1', 'X2'],
                class_names=['Clase 0', 'Clase 1'], ax=axes[idx])
        axes[idx].set_title(f'Árbol {idx+1} del Random Forest')
    plt.tight_layout()
    plt.show()

# Punto de entrada principal
if __name__ == "__main__":
    print("Generando visualizaciones...")
    # Generar todas las visualizaciones una por una
    plot_knn()
    plot_logistic_regression()
    plot_decision_tree()
    plot_random_forest()
