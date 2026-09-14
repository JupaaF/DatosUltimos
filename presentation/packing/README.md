# Packing de 500 esferas: densidad nominal final 58.5 %

Desde la raíz del proyecto:

```bash
.venv/bin/python presentation/packing/simulate.py
.venv/bin/manim -qh --fps 60 presentation/packing/scene.py PackingDemo
```

La diapositiva 3 de `presentation/presentation.py` reproduce estos mismos datos.
Para renderizar la presentación completa:

```bash
.venv/bin/manim -qh --fps 60 presentation/presentation.py Presentacion
```

## Datos

- `trajectory.npz`: animación de 5 segundos, 301 muestras a intervalos de 1/60 s.
- `trajectory_physics.npz`: tiempo físico a intervalos de 1/60 s; duración en `metadata.json`.
- `final_packing.csv`: columnas `id,radius,x,y,z`, una fila por esfera.
- `metadata.json`: parámetros, densidad nominal y comprobaciones físicas.
- `preview.png`: fotograma final renderizado con Manim.

Ambos NPZ contienen `radii` de forma `(500,)`, `positions` de forma `(frames,500,3)`,
`time`, `active_count`, `box_size` (dimensiones interiores finales),
`initial_box_size`, `box_height` (altura de la caja en cada muestra) y `lid_fraction`
(progreso del cierre lateral). La versión de animación añade `physical_time`.
Los índices de las partículas se mantienen durante toda la simulación.
Antes del nacimiento, las coordenadas son NaN: utiliza `active_count` para ocultarlas.
El radio es constante; no se duplica por fotograma.

```python
import numpy as np
with np.load('presentation/packing/trajectory.npz') as data:
    radii = data['radii']
    positions = data['positions']
    count = int(data['active_count'][150])
    centers = positions[150, :count]  # 2.5 s de animación
```

## Densidad y calendario

Coordenadas en metros: origen en una esquina inferior; Z vertical. Caja inicial de
3.4 × 3.4 × 3.2 m, radios uniformes entre 0.18 y 0.22 m, semilla 42.
La altura final se calcula como `sum(4*pi*r**3/3) / (0.585 * ancho * fondo)`.
La densidad nominal es el volumen de las esferas dividido por el volumen interior
final de la caja. Es una fracción global; el pequeño solapamiento de los contactos
blandos no se descuenta de la suma de volúmenes.

- 0.05–22 s físicos: nacimientos individuales a Z=4 m, sin solapamiento inicial.
- 22–26 s: asentamiento.
- 26–28 s: la tapa se desliza horizontalmente y cierra la caja.
- 28–42 s: la tapa física baja suavemente hasta una densidad nominal del 59 %, compactando el lecho.
- 42–55 s: reposo.
- 55–60 s: alivio suave de presión; la tapa sube unos 2.1 cm hasta la densidad final del 58.5 %.
- Desde 60 s: reposo. A partir de 65 s se comprueba cada 5 s si la velocidad máxima ha bajado de 0.025 m/s, con un límite de 90 s.

La caja se representa con paredes telescópicas: su borde superior acompaña a la tapa
al comprimir, por lo que `box_height` define el volumen interior en cada instante.
La reproducción acelera el tiempo por un factor `physical_seconds / 5`: pueden aparecer varias partículas
entre dos fotogramas, aunque cada nacimiento físico sea individual.

## Modelo y validación

DEM propio de esferas blandas: gravedad, integración semiimplícita a 3600 Hz,
resorte normal lineal de 50000 N/m, amortiguación y fricción tangencial viscosa limitada por Coulomb, con coeficiente de fricción 0.10.
La detección de contactos usa `scipy.spatial.cKDTree`. Sólo necesita NumPy y SciPy.
La tapa aplica fuerzas de contacto usando la velocidad relativa de la esfera respecto
al movimiento de la tapa. No se escalan artificialmente posiciones ni radios.
No incluye rotaciones ni memoria de fricción estática; sirve para ilustración,
no como modelo cuantitativo validado de un material.

El generador comprueba que las 500 esferas estén dentro de la caja, que la tapa pueda
cerrarse horizontalmente sin atravesarlas, velocidad final máxima inferior a 0.05 m/s
y penetración final entre esferas inferior a 0.01 m. Consulta los valores medidos en
`metadata.json`. La caja se dibuja con aristas, suelo y tapa teselados y ordenados en
profundidad para no superponer elementos traseros a las partículas en Cairo.

La skill reutilizable está en `skills/dem-packing-manim/SKILL.md`.
