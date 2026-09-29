# Series de stable packings

Los tres CSV tienen las mismas 37 columnas: las mediciones originales compartidas, más `series`. Se eliminó `packing_file` y no se incluyeron las columnas auxiliares exclusivas del archivo horizontal.

| Archivo | Series | Filas |
| --- | --- | ---: |
| `stable_packings_horizontal.csv` | `density_0.590` a `density_0.640`, en pasos de 0.005 | 220 |
| `stable_packings_vertical.csv` | `pressure_5000_pa`, `pressure_10000_pa`, `pressure_20000_pa`, `pressure_30000_pa`, `pressure_50000_pa`, `pressure_90000_pa`, `pressure_125000_pa`, `pressure_180000_pa`, `pressure_200000_pa` | 596 |
| `stable_packings_cyclic.csv` | `cyclic_1`, `cyclic_2` | 209 |

`series` identifica la serie experimental de cada fila. `checkpoint` se reinicia en cada serie.

## Unidades

- `p`, `q` y todos los componentes `sigma_*` están en Pa. Se verificó que `p = tr(sigma)/3`, que `q` coincide con el invariante desviador de `sigma` y que `q_over_p = q/p`.
- Las 140 filas horizontales que ya estaban en el archivo principal tenían `q` en kPa. Se multiplicó esa columna por 1000 para dejarla en Pa, igual que en todas las demás series. `q_over_p` ya era adimensional y no se modificó.
- `packing`, `q_over_p` y los componentes `Fabric_*` son adimensionales. `kd` y `a` conservan la escala de sus respectivos tensores `K_*` y `Fabric_*`.
- Los archivos de origen no especifican la unidad física de `time` ni de `K_*`/`kd`; se conservaron sus valores sin conversión.
