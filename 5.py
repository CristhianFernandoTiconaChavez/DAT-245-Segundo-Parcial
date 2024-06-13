import sqlite3

conn = sqlite3.connect('productos.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS productos (
    id INTEGER PRIMARY KEY,
    nombre TEXT,
    categoria TEXT,
    precio REAL,
    calificacion REAL
)
''')

productos = [
    ("Laptop A", "Electrónica", 999.99, 4.5),
    ("Laptop B", "Electrónica", 849.99, 4.0),
    ("Teléfono A", "Electrónica", 599.99, 4.3),
    ("Teléfono B", "Electrónica", 699.99, 4.6),
    ("Refrigerador A", "Electrodomésticos", 499.99, 4.2),
    ("Refrigerador B", "Electrodomésticos", 599.99, 4.8),
    ("Televisor A", "Electrónica", 399.99, 4.1),
    ("Televisor B", "Electrónica", 499.99, 4.7)
]

c.executemany('''
INSERT INTO productos (nombre, categoria, precio, calificacion)
VALUES (?, ?, ?, ?)
''', productos)

conn.commit()
conn.close()


import sqlite3

class AgenteInteligente:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()

    def buscar_productos(self, categoria=None, rango_precio=None, min_calificacion=None):
        query = "SELECT * FROM productos WHERE 1=1"
        params = []

        if categoria:
            query += " AND categoria = ?"
            params.append(categoria)

        if rango_precio:
            query += " AND precio BETWEEN ? AND ?"
            params.append(rango_precio[0])
            params.append(rango_precio[1])

        if min_calificacion:
            query += " AND calificacion >= ?"
            params.append(min_calificacion)

        self.c.execute(query, params)
        return self.c.fetchall()

    def comparar_productos(self, productos):
        print(f"{'ID':<3} {'Nombre':<20} {'Categoría':<15} {'Precio':<10} {'Calificación':<10}")
        print("="*60)
        for prod in productos:
            print(f"{prod[0]:<3} {prod[1]:<20} {prod[2]:<15} {prod[3]:<10.2f} {prod[4]:<10.1f}")

def main():
    agente = AgenteInteligente('productos.db')
    print("Bienvenido al agente inteligente de búsqueda comparativa")

    while True:
        print("\nOpciones de búsqueda:"
        )
        categoria = input("Categoría (dejar en blanco para todas): ")
        rango_precio = input("Rango de precio (min,max) (dejar en blanco para todas): ")
        min_calificacion = input("Calificación mínima (dejar en blanco para todas): ")

        rango_precio = tuple(map(float, rango_precio.split(','))) if rango_precio else None
        min_calificacion = float(min_calificacion) if min_calificacion else None

        productos = agente.buscar_productos(categoria, rango_precio, min_calificacion)
        agente.comparar_productos(productos)

        continuar = input("¿Desea realizar otra búsqueda? (s/n): ")
        if continuar.lower() != 's':
            break

if __name__ == "__main__":
    main()