import numpy as np

np.random.seed(42)
num_ciudades = 5
distancias = np.random.randint(10, 100, size=(num_ciudades, num_ciudades))

for i in range(num_ciudades):
    for j in range(i + 1, num_ciudades):
        distancias[j, i] = distancias[i, j]
    distancias[i, i] = 0

print("Matriz de distancias:")
print(distancias)

def costo(ruta, distancias):
    return sum(distancias[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1)) + distancias[ruta[-1], ruta[0]]

def solucion_inicial(num_ciudades):
    return list(np.random.permutation(num_ciudades))

def generar_vecino(ruta):
    vecino = ruta.copy()
    i, j = np.random.choice(len(ruta), 2, replace=False)
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

def simulated_annealing(distancias, temp_inicial, tasa_enfriamiento, iteraciones):
    ruta_actual = solucion_inicial(len(distancias))
    costo_actual = costo(ruta_actual, distancias)
    mejor_ruta = ruta_actual
    mejor_costo = costo_actual

    temperatura = temp_inicial

    for i in range(iteraciones):
        vecino = generar_vecino(ruta_actual)
        costo_vecino = costo(vecino, distancias)
        delta_costo = costo_vecino - costo_actual

        if delta_costo < 0 or np.random.rand() < np.exp(-delta_costo / temperatura):
            ruta_actual = vecino
            costo_actual = costo_vecino

            if costo_actual < mejor_costo:
                mejor_ruta = ruta_actual
                mejor_costo = costo_actual

        temperatura *= tasa_enfriamiento

    return mejor_ruta, mejor_costo

temp_inicial = 1000
tasa_enfriamiento = 0.99
iteraciones = 10000

mejor_ruta, mejor_costo = simulated_annealing(distancias, temp_inicial, tasa_enfriamiento, iteraciones)
print(f'Mejor ruta encontrada: {mejor_ruta}')
print(f'Costo de la mejor ruta: {mejor_costo}')
