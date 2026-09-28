# Histogramas de los packings

Cada PDF tiene seis páginas y representa 35 variables, con hasta seis histogramas por página. Se excluyen `series`, `time` y `checkpoint`. Se añade `K = (K_xx + K_yy + K_zz) / 3`, calculada solo para los gráficos. El eje X usa entre 60 y 120 intervalos por variable para mostrar más detalle.

| Archivo | Datos |
| --- | --- |
| `histogramas_horizontal.pdf` | Horizontal, 200 filas |
| `histogramas_vertical.pdf` | Vertical, 596 filas |
| `histogramas_cyclic_1.pdf` | Cíclico 1, 85 filas |
| `histogramas_cyclic_2.pdf` | Cíclico 2, 124 filas |
| `histogramas_comparacion.pdf` | Un único histograma por variable con las 1005 filas reunidas |

El eje vertical indica frecuencia absoluta en todos los PDF. `comparacion_vista_previa.png` muestra la primera página del histograma conjunto.

Para regenerarlos desde la raíz del proyecto:

```bash
.venv/bin/python -m src.plot_packings_histograms
```
