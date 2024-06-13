import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

tamaño_entrada = X_entrenamiento.shape[1]
tamaño_oculto = 5
tamaño_salida = y_entrenamiento.shape[1]

np.random.seed(42)
W1 = np.random.randn(tamaño_entrada, tamaño_oculto)
b1 = np.zeros((1, tamaño_oculto))
W2 = np.random.randn(tamaño_oculto, tamaño_salida)
b2 = np.zeros((1, tamaño_salida))

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def derivada_sigmoide(z):
    return z * (1 - z)

def propagacion_adelante(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoide(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoide(z2)
    return a1, a2

def calcular_costo(y_verdadero, y_predicho):
    m = y_verdadero.shape[0]
    costo = -np.sum(y_verdadero * np.log(y_predicho) + (1 - y_verdadero) * np.log(1 - y_predicho)) / m
    return costo

def retropropagacion(X, y, a1, a2):
    m = y.shape[0]
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, W2.T) * derivada_sigmoide(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def actualizar_parametros(dW1, db1, dW2, db2, tasa_aprendizaje):
    global W1, b1, W2, b2
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2

tasa_aprendizaje = 0.4
epocas = 1000
costos = []
umbral_convergencia = 1e-5

epoca_convergencia = epocas 

for epoca in range(epocas):
    a1, a2 = propagacion_adelante(X_entrenamiento)
    
    costo = calcular_costo(y_entrenamiento, a2)
    costos.append(costo)
    
    dW1, db1, dW2, db2 = retropropagacion(X_entrenamiento, y_entrenamiento, a1, a2)
    
    actualizar_parametros(dW1, db1, dW2, db2, tasa_aprendizaje)
    
    if epoca % 100 == 0:
        print(f"Época {epoca}, costo: {costo}")
    
    if epoca > 0 and abs(costos[-1] - costos[-2]) < umbral_convergencia:
        epoca_convergencia = epoca
        print(f"Convergencia alcanzada en la época {epoca}")
        break

print(f"Última época, costo: {costos[-1]}")
print(f"Número de épocas requeridas para la convergencia: {epoca_convergencia}")
