"""
test_main.py
────────────
Unit tests para main.py — adherencia de excavación.
Corre con: pytest test_main.py -v

No requiere servidor uvicorn ni conexión a internet.
"""

import pytest
import math
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from main import (
    construir_solidos,
    calcular_adherencia,
    limpiar_json,
    anillos,
    rotar,
)


# ── limpiar_json ──────────────────────────────────────────────────────────────

def test_limpiar_json_nan():
    assert limpiar_json(float('nan')) is None

def test_limpiar_json_inf():
    assert limpiar_json(float('inf')) is None

def test_limpiar_json_numero_normal():
    assert limpiar_json(3.14) == 3.14

def test_limpiar_json_entero():
    assert limpiar_json(5) == 5

def test_limpiar_json_string():
    assert limpiar_json("hola") == "hola"

def test_limpiar_json_lista_con_nan():
    resultado = limpiar_json([1.0, float('nan'), 3.0])
    assert resultado == [1.0, None, 3.0]

def test_limpiar_json_dict_con_nan():
    resultado = limpiar_json({'a': 1.0, 'b': float('nan')})
    assert resultado == {'a': 1.0, 'b': None}

def test_limpiar_json_anidado():
    resultado = limpiar_json({'tabla': [{'adherencia': float('nan')}]})
    assert resultado == {'tabla': [{'adherencia': None}]}


# ── calcular_adherencia ───────────────────────────────────────────────────────

def test_adherencia_sin_puntos():
    """Sin puntos debe retornar None, no NaN."""
    poly = Polygon([(0,0),(100,0),(100,100),(0,100)])
    resultado = calcular_adherencia(poly, [])
    assert resultado is None

def test_adherencia_todos_dentro():
    """Puntos claramente dentro del polígono → 100%."""
    poly = Polygon([(0,0),(1000,0),(1000,1000),(0,1000)])
    puntos = [[500,500], [400,400], [300,600]]
    resultado = calcular_adherencia(poly, puntos)
    assert resultado == 100.0

def test_adherencia_todos_fuera():
    """Puntos claramente fuera del polígono → 0%."""
    poly = Polygon([(0,0),(10,0),(10,10),(0,10)])
    puntos = [[500,500], [400,400]]
    resultado = calcular_adherencia(poly, puntos)
    assert resultado == 0.0

def test_adherencia_mitad():
    """La mitad de los puntos dentro → ~50%."""
    poly = Polygon([(0,0),(100,0),(100,100),(0,100)])
    # 2 dentro, 2 fuera (lejos del borde para no caer en el buffer)
    puntos = [[50,50], [60,60], [500,500], [600,600]]
    resultado = calcular_adherencia(poly, puntos)
    assert resultado == 50.0

def test_adherencia_buffer():
    """Punto justo fuera del polígono pero dentro del buffer de 5m → dentro."""
    poly = Polygon([(0,0),(100,0),(100,100),(0,100)])
    # Punto a 3m fuera del borde (buffer=5m, debería quedar dentro)
    puntos = [[103, 50]]
    resultado = calcular_adherencia(poly, puntos)
    assert resultado == 100.0

def test_adherencia_retorna_float():
    poly = Polygon([(0,0),(100,0),(100,100),(0,100)])
    puntos = [[50,50]]
    resultado = calcular_adherencia(poly, puntos)
    assert isinstance(resultado, float)


# ── construir_solidos ─────────────────────────────────────────────────────────

POLS_TEST = [
    {'fecha': '2026-07-01', 'pala': 'S01', 'fase': 'F06',
     'vertices': [(0,0),(100,0),(100,100),(0,100),(0,0)]},
    {'fecha': '2026-07-05', 'pala': 'S01', 'fase': 'F06',
     'vertices': [(200,200),(300,200),(300,300),(200,300),(200,200)]},
    {'fecha': '2026-07-10', 'pala': 'S06', 'fase': 'F06',
     'vertices': [(500,500),(600,500),(600,600),(500,600),(500,500)]},
]

def test_construir_solidos_incluye_fecha_limite():
    """Polígonos hasta fecha límite inclusive deben incluirse."""
    solidos = construir_solidos(POLS_TEST, pd.Timestamp('2026-07-10').date())
    assert 'S01' in solidos
    assert 'S06' in solidos

def test_construir_solidos_excluye_fecha_posterior():
    """Polígonos después de la fecha límite no deben incluirse."""
    solidos = construir_solidos(POLS_TEST, pd.Timestamp('2026-07-05').date())
    assert 'S01' in solidos
    assert 'S06' not in solidos  # S06 es del 10/07

def test_construir_solidos_acumula_poligonos():
    """El sólido de S01 debe ser la unión de sus 2 polígonos."""
    solidos = construir_solidos(POLS_TEST, pd.Timestamp('2026-07-10').date())
    assert 'S01' in solidos
    # El área del union debe ser mayor que cada polígono individual
    area_union = solidos['S01'].area
    assert area_union > 100 * 100  # mayor que un solo polígono de 100x100

def test_construir_solidos_sin_poligonos():
    """Si no hay polígonos para la fecha, retorna dict vacío."""
    solidos = construir_solidos(POLS_TEST, pd.Timestamp('2026-06-01').date())
    assert solidos == {}

def test_construir_solidos_geometria_valida():
    """Los sólidos generados deben ser geometrías válidas."""
    solidos = construir_solidos(POLS_TEST, pd.Timestamp('2026-07-10').date())
    for pala, geom in solidos.items():
        assert geom.is_valid, f"Geometría inválida para {pala}"
        assert not geom.is_empty, f"Geometría vacía para {pala}"


# ── anillos ───────────────────────────────────────────────────────────────────

def test_anillos_poligono_simple():
    """Un polígono simple debe devolver 1 anillo (el exterior)."""
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    resultado = anillos(poly)
    assert len(resultado) == 1

def test_anillos_multipoligono():
    """Un MultiPolygon debe devolver múltiples anillos."""
    from shapely.geometry import MultiPolygon
    p1 = Polygon([(0,0),(1,0),(1,1),(0,1)])
    p2 = Polygon([(5,5),(6,5),(6,6),(5,6)])
    multi = MultiPolygon([p1, p2])
    resultado = anillos(multi)
    assert len(resultado) == 2

def test_anillos_devuelve_listas():
    """Cada anillo debe ser una lista de coordenadas."""
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    resultado = anillos(poly)
    assert isinstance(resultado[0], list)
    assert all(isinstance(c, (list, tuple)) for c in resultado[0])


# ── rotar ─────────────────────────────────────────────────────────────────────

def test_rotar_devuelve_array():
    pts = [[409716.0, 7496728.0]]
    resultado = rotar(pts)
    assert resultado.shape == (1, 2)

def test_rotar_centroide_se_mantiene():
    """El centroide (CX, CY) rotado sobre sí mismo no cambia."""
    from main import CX, CY
    pts = [[CX, CY]]
    resultado = rotar(pts)
    assert abs(resultado[0,0] - CX) < 1e-6
    assert abs(resultado[0,1] - CY) < 1e-6

def test_rotar_es_invertible():
    """Rotar -45° y luego +45° debe devolver el punto original."""
    import math
    from main import CX, CY, ANGULO
    pts = np.array([[CX + 100, CY + 200]])
    
    # Rotar con el ángulo original
    r1 = rotar(pts)
    
    # Rotar de vuelta (ángulo inverso)
    rad = math.radians(-ANGULO)
    dx = r1[:,0] - CX
    dy = r1[:,1] - CY
    x_inv = CX + dx * math.cos(rad) - dy * math.sin(rad)
    y_inv = CY + dx * math.sin(rad) + dy * math.cos(rad)
    
    assert abs(x_inv[0] - pts[0,0]) < 1e-6
    assert abs(y_inv[0] - pts[0,1]) < 1e-6
