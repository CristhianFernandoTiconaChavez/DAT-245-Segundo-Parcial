import numpy as np

np.random.seed(0)
num_ciudades = 5
ciudades = np.random.rand(num_ciudades, 2) 

def distancia(ciudad1, ciudad2):
    return np.linalg.norm(ciudad1 - ciudad2)

def distancia_total(ruta, ciudades):
    dist = 0
    for i in range(len(ruta) - 1):
        dist += distancia(ciudades[ruta[i]], ciudades[ruta[i+1]])
    dist += distancia(ciudades[ruta[-1]], ciudades[ruta[0]]) 
    return dist

def generar_vecino(ruta):
    nueva_ruta = ruta.copy()
    idx1, idx2 = np.random.choice(len(ruta), size=2, replace=False)
    nueva_ruta[idx1], nueva_ruta[idx2] = nueva_ruta[idx2], nueva_ruta[idx1]
    return nueva_ruta

def recocido_simulado_tsp(ciudades, temp_inicial, tasa_enfriamiento, max_iteraciones):
    num_ciudades = len(ciudades)
    ruta_actual = np.random.permutation(num_ciudades)
    costo_actual = distancia_total(ruta_actual, ciudades)
    
    mejor_ruta = ruta_actual
    mejor_costo = costo_actual
    
    temp = temp_inicial
    
    for iteracion in range(max_iteraciones):
        nueva_ruta = generar_vecino(ruta_actual)
        nuevo_costo = distancia_total(nueva_ruta, ciudades)
        
        if nuevo_costo < costo_actual or np.exp((costo_actual - nuevo_costo) / temp) > np.random.rand():
            ruta_actual = nueva_ruta
            costo_actual = nuevo_costo
        
        if costo_actual < mejor_costo:
            mejor_ruta = ruta_actual
            mejor_costo = costo_actual
        
        temp *= tasa_enfriamiento
        
        if temp < 1e-3:
            break
    
    return mejor_ruta, mejor_costo

temp_inicial = 1000
tasa_enfriamiento = 0.95
max_iteraciones = 1000

mejor_ruta, mejor_costo = recocido_simulado_tsp(ciudades, temp_inicial, tasa_enfriamiento, max_iteraciones)

print("Mejor ruta encontrada:")
print(mejor_ruta)
print("Mejor costo encontrado:")
print(mejor_costo)
