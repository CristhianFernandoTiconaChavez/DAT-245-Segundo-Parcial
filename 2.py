import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

datos = load_iris()
X = datos['data']
y = datos['target']

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

np.random.seed(42)
tamaño_entrada = X_entrenamiento.shape[1]
tamaño_oculto1 = 5
tamaño_oculto2 = 5
tamaño_salida = y_entrenamiento.shape[1]
pesos_entrada_oculto1 = np.random.rand(tamaño_entrada, tamaño_oculto1) - 0.5
pesos_oculto1_oculto2 = np.random.rand(tamaño_oculto1, tamaño_oculto2) - 0.5
pesos_oculto2_salida = np.random.rand(tamaño_oculto2, tamaño_salida) - 0.5

def escalon(x):
    return np.where(x >= 0, 1, 0)

tasa_aprendizaje = 0.2
epocas = 1000

for epoca in range(epocas):
    entrada_oculta1 = np.dot(X_entrenamiento, pesos_entrada_oculto1)
    salida_oculta1 = escalon(entrada_oculta1)

    entrada_oculta2 = np.dot(salida_oculta1, pesos_oculto1_oculto2)
    salida_oculta2 = escalon(entrada_oculta2)

    entrada_final = np.dot(salida_oculta2, pesos_oculto2_salida)
    salida_final = escalon(entrada_final)

    error = y_entrenamiento - salida_final

    d_salida = error
    error_oculta2 = d_salida.dot(pesos_oculto2_salida.T)
    d_oculta2 = error_oculta2

    error_oculta1 = d_oculta2.dot(pesos_oculto1_oculto2.T)
    d_oculta1 = error_oculta1

    pesos_oculto2_salida += salida_oculta2.T.dot(d_salida) * tasa_aprendizaje
    pesos_oculto1_oculto2 += salida_oculta1.T.dot(d_oculta2) * tasa_aprendizaje
    pesos_entrada_oculto1 += X_entrenamiento.T.dot(d_oculta1) * tasa_aprendizaje

    if epoca % 100 == 0:
        error_cuadratico_medio = np.mean(np.square(error))
        print(f'Época {epoca}, Pérdida: {error_cuadratico_medio}')

print(f'Número de épocas requeridas: {epocas}')

entrada_oculta1_prueba = np.dot(X_prueba, pesos_entrada_oculto1)
salida_oculta1_prueba = escalon(entrada_oculta1_prueba)

entrada_oculta2_prueba = np.dot(salida_oculta1_prueba, pesos_oculto1_oculto2)
salida_oculta2_prueba = escalon(entrada_oculta2_prueba)

entrada_final_prueba = np.dot(salida_oculta2_prueba, pesos_oculto2_salida)
salida_final_prueba = escalon(entrada_final_prueba)

error_prueba = y_prueba - salida_final_prueba
error_cuadratico_medio_prueba = np.mean(np.square(error_prueba))
print(f'Error en el conjunto de prueba: {error_cuadratico_medio_prueba}')