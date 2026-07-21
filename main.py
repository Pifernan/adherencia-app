from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import math, io, json
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Transformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CX, CY   = 409716.0, 7496728.0
ANGULO   = -45
BUFFER_M = 5.0

def rotar(pts):
    arr = np.array(pts)
    rad = math.radians(ANGULO)
    dx, dy = arr[:,0]-CX, arr[:,1]-CY
    return np.column_stack([
        CX + dx*math.cos(rad) - dy*math.sin(rad),
        CY + dx*math.sin(rad) + dy*math.cos(rad)
    ])

def construir_solidos(poligonos, fecha_limite):
    grupos = {}
    for pol in poligonos:
        if pd.Timestamp(pol['fecha']).date() > fecha_limite:
            continue
        pala = pol['pala']
        if pala not in grupos:
            grupos[pala] = []
        grupos[pala].append(pol['vertices'])

    solidos = {}
    for pala, lista_verts in grupos.items():
        geoms = []
        for verts in lista_verts:
            try:
                g = Polygon(verts)
                if not g.is_valid:
                    g = g.buffer(0)
                if not g.is_empty:
                    geoms.append(g)
            except:
                pass
        if geoms:
            union = unary_union(geoms)
            if not union.is_empty:
                solidos[pala] = union
    return solidos

def anillos(geom):
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for poly in polys:
        out.append(np.array(poly.exterior.coords).tolist())
        for h in poly.interiors:
            out.append(np.array(h.coords).tolist())
    return out

def calcular_adherencia(geom, xy):
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        return None
    geom_buf = geom.buffer(BUFFER_M)
    x, y = xy[:,0], xy[:,1]
    dentro = shapely.contains_xy(geom_buf, x, y)
    return round(100.0 * dentro.sum() / len(xy), 1)

def limpiar_json(obj):
    """Elimina NaN e inf recursivamente para que JSON no falle."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [limpiar_json(i) for i in obj]
    if isinstance(obj, dict):
        return {k: limpiar_json(v) for k, v in obj.items()}
    return obj

@app.post("/analizar")
async def analizar(
    csv_file:       UploadFile = File(...),
    poligonos_file: UploadFile = File(...),
    topo_file:      UploadFile = File(None),
    fecha_inicio:   str        = Form(...),
    fecha_fin:      str        = Form(...),
):
    try:
        fecha_inicio_dt = pd.Timestamp(fecha_inicio).date()
        fecha_fin_dt    = pd.Timestamp(fecha_fin).date()
        fecha_limite    = fecha_fin_dt  # polígonos acumulados hasta fecha fin

        # ── polígonos ──────────────────────────────────────────────────────
        data = json.loads(await poligonos_file.read())
        poligonos = data['poligonos']
        solidos = construir_solidos(poligonos, fecha_limite)

        if not solidos:
            return JSONResponse({"error": f"Sin polígonos hasta {fecha_fin}"}, 400)

        # ── topografía ─────────────────────────────────────────────────────
        segs_topo = []
        if topo_file:
            topo = json.loads(await topo_file.read())
            ang_inv = math.radians(ANGULO)
            cos_a, sin_a = math.cos(ang_inv), math.sin(ang_inv)
            for linea in topo.get('lineas', []):
                pts = np.array(linea, dtype=float)
                # Coordenadas UTM reales — solo rotar para alinear con los sólidos
                seg = rotar(pts)
                segs_topo.append({'x': seg[:,0].tolist(), 'y': seg[:,1].tolist()})

        # ── puntos CSV ─────────────────────────────────────────────────────
        raw = await csv_file.read()
        sample = raw[:2000].decode('utf-8-sig', errors='ignore')
        sep = ',' if sample.count(',') > sample.count(';') else ';'
        puntos = pd.read_csv(io.BytesIO(raw), sep=sep, encoding='utf-8-sig')
        puntos = puntos[puntos['longitude'].notna() & puntos['latitude'].notna()].copy()

        # ── parseo robusto de time_tripped ────────────────────────────────
        # Origen real en BD: "2026-07-10 00:05:54.000".
        # Para evitar que Excel destruya la columna, se recomienda exportar
        # el tiempo como epoch (entero de segundos) o en un formato que Excel
        # no reconozca como fecha (ej: 2026-07-10_00-05-54). Este parser acepta
        # cualquiera de esas variantes y detecta si el archivo llegó corrupto.
        def parse_tiempo(serie_raw):
            s = serie_raw.astype(str).str.strip()

            # --- Guardia: detectar corrupción de Excel ---
            # Si Excel interpretó los timestamps como duración, quedan como
            # "MM:SS.0" (sin fecha ni hora recuperable). No hay forma de
            # reconstruir la fecha: se aborta con un mensaje claro.
            corrupto = s.str.match(r'^\d{1,2}:\d{2}\.\d+$')
            if corrupto.mean() > 0.5:
                raise ValueError(
                    "El CSV llegó con la columna de tiempo corrompida por Excel "
                    "(valores tipo 'MM:SS.0', sin fecha). La información original "
                    "no es recuperable. Exporta el tiempo como epoch (entero de "
                    "segundos) o en formato AAAA-MM-DD_HH-MM-SS para que Excel no "
                    "lo altere."
                )

            # --- Caso epoch: columna puramente numérica ---
            num = pd.to_numeric(s, errors='coerce')
            if num.notna().mean() > 0.95:
                # segundos (~1.7e9) o milisegundos (~1.7e12) según magnitud
                unidad = 'ms' if num.dropna().median() > 1e11 else 's'
                return pd.to_datetime(num, unit=unidad, utc=True).dt.tz_localize(None)

            # --- Caso formato con guiones bajos: AAAA-MM-DD_HH-MM-SS ---
            guion_bajo = s.str.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')
            if guion_bajo.all():
                s = s.str.replace('_', ' ', regex=False)
                s = s.str.replace(r'(\d{2}) (\d{2})-(\d{2})-(\d{2})',
                                  r'\1 \2:\3:\4', regex=True)
                return pd.to_datetime(s, format='mixed')

            # --- Caso ISO inequívoco: AAAA-MM-DD ... ---
            if s.str.match(r'^\d{4}-\d{2}-\d{2}').all():
                return pd.to_datetime(s, format='mixed')

            # --- Último recurso: formatos con "/" (día primero, config CL) ---
            return pd.to_datetime(s, dayfirst=True, format='mixed')

        puntos['time_tripped'] = parse_tiempo(puntos['time_tripped'])

        # Filtro temprano por fecha para reducir memoria
        puntos['_date_raw'] = puntos['time_tripped'].dt.date
        puntos = puntos[puntos['_date_raw'].between(
            fecha_inicio_dt - pd.Timedelta(days=1),
            fecha_fin_dt    + pd.Timedelta(days=1)
        )].drop(columns='_date_raw').copy()

        puntos['lon_wgs84'] = puntos['longitude'] / 3_600_000
        puntos['lat_wgs84'] = puntos['latitude']  / 3_600_000
        t = Transformer.from_crs(4326, 24879, always_xy=True)
        puntos['x'], puntos['y'] = t.transform(
            puntos['lon_wgs84'].values, puntos['lat_wgs84'].values)

        puntos['time_tripped'] = (
            puntos['time_tripped']
            .dt.tz_localize('UTC')
            .dt.tz_convert('America/Santiago')
        )
        hora      = puntos['time_tripped'].dt.hour
        fecha_col = puntos['time_tripped'].dt.date

        puntos['date_op'] = np.select(
            [hora < 8, (hora >= 8) & (hora < 20), hora >= 20],
            [fecha_col - pd.Timedelta(days=1), fecha_col, fecha_col],
            default=pd.NaT)
        puntos['date_op'] = pd.to_datetime(puntos['date_op'])

        if 'elevation' in puntos.columns:
            puntos = puntos.rename(columns={'elevation': 'z'})
        puntos['fase'] = ('F0' +
            puntos['fase'].str.extract(r'FASE\s*(\d)', expand=False).fillna(''))
        puntos = puntos[puntos['shovel'] != 'L01'].copy()

        if 'SH01' in puntos['shovel'].values and 'SH02' in puntos['shovel'].values:
            sh01_y_min = puntos[puntos['shovel'] == 'SH01']['y'].min()
            puntos = puntos[
                ~((puntos['shovel'] == 'SH02') & (puntos['y'] > sh01_y_min))
            ].copy()

        puntos_clean = puntos[
            (puntos['date_op'] >= pd.Timestamp(fecha_inicio)) &
            (puntos['date_op'] <= pd.Timestamp(fecha_fin))
        ].reset_index(drop=True)

        # ── adherencia ─────────────────────────────────────────────────────
        _tab10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        palas = sorted(set(solidos.keys()) | set(puntos_clean['shovel'].dropna()))
        color_pala = {p: _tab10[i % 10] for i, p in enumerate(sorted(palas))}

        tabla = []
        for pala, geom in solidos.items():
            sub = puntos_clean[puntos_clean['shovel'] == pala]
            adh = calcular_adherencia(geom, sub[['x','y']].to_numpy())
            tabla.append({
                'pala': pala,
                'n_poligonos': len([p for p in poligonos
                                    if p['pala'] == pala
                                    and pd.Timestamp(p['fecha']).date() <= fecha_limite]),
                'n_puntos': len(sub),
                'adherencia': adh,
            })
        tabla.sort(key=lambda r: r['pala'])

        # ── datos gráfico ──────────────────────────────────────────────────
        solidos_plot = []
        for pala, geom in solidos.items():
            xs, ys = [], []
            for anillo in anillos(geom):
                r = rotar(np.array(anillo))
                xs.extend(r[:,0].tolist() + [None])
                ys.extend(r[:,1].tolist() + [None])
            solidos_plot.append({
                'pala': pala, 'xs': xs, 'ys': ys,
                'color': color_pala.get(pala, '#888')
            })

        puntos_plot = []
        for pala in sorted(palas):
            sub = puntos_clean[puntos_clean['shovel'] == pala]
            if sub.empty:
                continue
            pts_r = rotar(sub[['x','y']].to_numpy())
            puntos_plot.append({
                'pala': pala,
                'x': pts_r[:,0].tolist(),
                'y': pts_r[:,1].tolist(),
                'color': color_pala.get(pala, '#888')
            })

        resultado = limpiar_json({
            'tabla':          tabla,
            'solidos':        solidos_plot,
            'puntos':         puntos_plot,
            'topo_segs':      segs_topo,
            'fecha_inicio':   fecha_inicio,
            'fecha_fin':      fecha_fin,
            'n_puntos_total': len(puntos_clean),
        })

        return resultado

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "detalle": traceback.format_exc()}, 500)
