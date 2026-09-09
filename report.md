# RBF architecture complexity study

Campaign: `DATASET`
Levels: `N174`
Total rows analyzed: `174`

Target: `log10(K)`.
Feature scaling: `none`.
KN uses a reproducible random `r ~ U(0,1)` with seeds N174: `6216`.

## RBF parameters

- `kernel`: radial basis shape used for interpolation/smoothing.
- `degree`: polynomial degree added to the RBF; it controls the global trend component.
- `smoothing`: regularization; larger values smooth more, smaller values interpolate more closely.
- `neighbors`: if `None`, each prediction uses all training points; if integer, predictions use local nearest neighbors.

## RBF configurations

| Model | kernel | degree | smoothing | neighbors | Intent |
| --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | Muy suave/simple, tendencia global |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | Suave global |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | Configuracion cercana a la actual |
| rbf_4 | cubic | 1 | 1e-08 | 40 | Mas flexible/local |
| rbf_5 | quintic | 2 | 1e-10 | 25 | Mas compleja/local, mayor riesgo de sobreajuste |

## Best architecture by protocol

Selection criterion: lowest cross-validated RMSE on `log10(K)`.

| Level | Protocol | Definition | RBF | 1 - R2 | RMSE | MAE | kernel | degree | smoothing | neighbors | scaled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N174 | K0 | K(p) | rbf_2 | 2.29e-03 | 0.01604 | 0.01388 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K1 | K(p,alpha) | rbf_2 | 2.30e-04 | 0.00508 | 0.00274 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K2 | K(p,MCN) | rbf_2 | 9.10e-05 | 0.00319 | 0.00173 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K3 | K(p,alpha,qnorm) | rbf_4 | 5.21e-05 | 0.00242 | 0.00152 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K4 | K(p,alpha,MCN) | rbf_4 | 1.88e-05 | 0.00145 | 0.00057 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K5 | K(p,MCN,qnorm) | rbf_3 | 4.52e-05 | 0.00225 | 0.00138 | thin_plate_spline | 1 | 1e-06 | 60 | no |
| N174 | K6 | K(p,alpha,q[kPa]) | rbf_4 | 2.80e-05 | 0.00177 | 0.00083 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K7 | K(alpha,q[kPa]) | rbf_1 | 7.80e-02 | 0.09356 | 0.06382 | linear | 0 | 0.01 | None | no |
| N174 | K8 | K(p,alpha,q[K]) | rbf_5 | 2.79e-05 | 0.00177 | 0.00082 | quintic | 2 | 1e-10 | 25 | no |
| N174 | K9 | K(p,alpha,MCN,qnorm) | rbf_5 | 1.68e-05 | 0.00137 | 0.00058 | quintic | 2 | 1e-10 | 25 | no |
| N174 | K10 | K(p,alpha,MCN,q[K]) | rbf_4 | 1.99e-05 | 0.00150 | 0.00058 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K11 | K(p,qnorm) | rbf_1 | 1.02e-03 | 0.01070 | 0.00913 | linear | 0 | 0.01 | None | no |
| N174 | K12 | K(p,q[K]) | rbf_4 | 1.05e-03 | 0.01085 | 0.00944 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K13 | K(p,a) | rbf_2 | 1.91e-03 | 0.01462 | 0.01200 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K14 | K(p,alpha,a) | rbf_5 | 1.04e-04 | 0.00341 | 0.00172 | quintic | 2 | 1e-10 | 25 | no |
| N174 | K15 | K(p,MCN,a) | rbf_2 | 1.10e-04 | 0.00351 | 0.00230 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K16 | K(p,alpha,MCN,a) | rbf_4 | 2.15e-05 | 0.00155 | 0.00063 | cubic | 1 | 1e-08 | 40 | no |
| N174 | K19 | K(p,MCN,q[kPa]) | rbf_2 | 3.23e-05 | 0.00190 | 0.00101 | thin_plate_spline | 1 | 0.001 | None | no |
| N174 | K20 | K(p,alpha,MCN,q[kPa]) | rbf_4 | 1.78e-05 | 0.00141 | 0.00055 | cubic | 1 | 1e-08 | 40 | no |
| N174 | KN | K(p,alpha,r) | rbf_2 | 2.55e-04 | 0.00535 | 0.00355 | thin_plate_spline | 1 | 0.001 | None | no |

## Metrics by protocol

## Level N174

### K0 = K(p)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.29e-03 | 0.01604 | 0.01388 | 0.032001 | 0.098528 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.29e-03 | 0.01604 | 0.01388 | 0.032009 | 0.098549 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.29e-03 | 0.01604 | 0.01387 | 0.031989 | 0.098562 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.30e-03 | 0.01605 | 0.01388 | 0.032011 | 0.098629 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.31e-03 | 0.01609 | 0.01393 | 0.032131 | 0.099654 |

### K1 = K(p,alpha)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 8.72e-04 | 0.00989 | 0.00590 | 0.013581 | 0.12563 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.30e-04 | 0.00508 | 0.00274 | 0.0063096 | 0.052621 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.88e-04 | 0.00569 | 0.00294 | 0.0067707 | 0.066598 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 6.97e-04 | 0.00884 | 0.00400 | 0.0091712 | 0.13391 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 3.39e-03 | 0.01950 | 0.00708 | 0.015693 | 0.38456 |

### K2 = K(p,MCN)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.99e-04 | 0.00579 | 0.00344 | 0.0079043 | 0.076021 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 9.10e-05 | 0.00319 | 0.00173 | 0.0039706 | 0.033931 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.04e-04 | 0.00342 | 0.00181 | 0.004163 | 0.036617 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.04e-04 | 0.00478 | 0.00243 | 0.0055864 | 0.061229 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 1.04e-03 | 0.01079 | 0.00483 | 0.011274 | 0.13788 |

### K3 = K(p,alpha,qnorm)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 8.34e-04 | 0.00967 | 0.00564 | 0.012961 | 0.1382 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.11e-04 | 0.00353 | 0.00207 | 0.0047773 | 0.054176 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.31e-04 | 0.00384 | 0.00224 | 0.0051772 | 0.062657 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 5.21e-05 | 0.00242 | 0.00152 | 0.0035104 | 0.028052 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 5.25e-05 | 0.00243 | 0.00097 | 0.0022351 | 0.036499 |

### K4 = K(p,alpha,MCN)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 7.52e-04 | 0.00919 | 0.00464 | 0.01065 | 0.12313 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.37e-05 | 0.00163 | 0.00064 | 0.0014803 | 0.034995 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.30e-05 | 0.00161 | 0.00064 | 0.0014894 | 0.034576 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.88e-05 | 0.00145 | 0.00057 | 0.001324 | 0.033326 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.25e-05 | 0.00159 | 0.00056 | 0.0012928 | 0.039465 |

### K5 = K(p,MCN,qnorm)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.86e-04 | 0.00567 | 0.00327 | 0.0075112 | 0.081937 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 5.26e-05 | 0.00243 | 0.00160 | 0.0036888 | 0.03937 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 4.52e-05 | 0.00225 | 0.00138 | 0.0031718 | 0.037059 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 4.77e-05 | 0.00231 | 0.00125 | 0.0028861 | 0.03425 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 6.02e-05 | 0.00260 | 0.00108 | 0.0024937 | 0.05664 |

### K6 = K(p,alpha,q[kPa])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.04e-03 | 0.01513 | 0.00993 | 0.022836 | 0.14493 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 3.41e-05 | 0.00196 | 0.00094 | 0.0021761 | 0.031613 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 3.38e-05 | 0.00195 | 0.00093 | 0.0021529 | 0.031376 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.80e-05 | 0.00177 | 0.00083 | 0.001918 | 0.031795 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 7.72e-05 | 0.00294 | 0.00109 | 0.0025115 | 0.052615 |

### K7 = K(alpha,q[kPa])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 7.80e-02 | 0.09356 | 0.06382 | 0.15088 | 1.0213 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 9.49e-02 | 0.10317 | 0.06691 | 0.16047 | 1.2355 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.15e-01 | 0.11340 | 0.07192 | 0.17416 | 1.4329 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.37e-01 | 0.16319 | 0.09309 | 0.25038 | 6.1828 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 3.30e+00 | 0.60879 | 0.26287 | 75.82 | 11844 |

### K8 = K(p,alpha,q[K])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 7.37e-04 | 0.00909 | 0.00557 | 0.012799 | 0.11729 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 3.42e-04 | 0.00619 | 0.00346 | 0.0079651 | 0.069235 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 3.37e-04 | 0.00615 | 0.00333 | 0.0076674 | 0.057291 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.64e-04 | 0.00544 | 0.00327 | 0.0075265 | 0.04983 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.79e-05 | 0.00177 | 0.00082 | 0.0018863 | 0.025156 |

### K9 = K(p,alpha,MCN,qnorm)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 6.67e-04 | 0.00865 | 0.00452 | 0.010354 | 0.11712 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.04e-05 | 0.00151 | 0.00066 | 0.0015139 | 0.032777 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.98e-05 | 0.00149 | 0.00062 | 0.001432 | 0.032639 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.71e-05 | 0.00138 | 0.00057 | 0.0013148 | 0.029364 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 1.68e-05 | 0.00137 | 0.00058 | 0.0013342 | 0.025085 |

### K10 = K(p,alpha,MCN,q[K])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 6.83e-04 | 0.00875 | 0.00450 | 0.010311 | 0.13106 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.50e-05 | 0.00167 | 0.00066 | 0.0015163 | 0.035529 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.36e-05 | 0.00163 | 0.00064 | 0.0014834 | 0.03518 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.99e-05 | 0.00150 | 0.00058 | 0.0013301 | 0.034478 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 4.20e-05 | 0.00217 | 0.00059 | 0.0013737 | 0.059351 |

### K11 = K(p,qnorm)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 1.02e-03 | 0.01070 | 0.00913 | 0.021081 | 0.10982 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.03e-03 | 0.01074 | 0.00908 | 0.020965 | 0.10518 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.76e-03 | 0.01406 | 0.01193 | 0.027616 | 0.1213 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.36e-03 | 0.01626 | 0.01354 | 0.031324 | 0.13677 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.06e-03 | 0.01522 | 0.01205 | 0.027734 | 0.12752 |

### K12 = K(p,q[K])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.29e-03 | 0.01604 | 0.01385 | 0.031969 | 0.099832 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.58e-03 | 0.01331 | 0.01113 | 0.025678 | 0.10362 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.07e-03 | 0.01095 | 0.00945 | 0.021782 | 0.10961 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.05e-03 | 0.01085 | 0.00944 | 0.021767 | 0.10955 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 1.12e-03 | 0.01121 | 0.00949 | 0.021932 | 0.10287 |

### K13 = K(p,a)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.14e-03 | 0.01550 | 0.01314 | 0.030345 | 0.099394 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.91e-03 | 0.01462 | 0.01200 | 0.027615 | 0.1002 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.37e-03 | 0.01629 | 0.01364 | 0.031559 | 0.1021 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.47e-03 | 0.01666 | 0.01398 | 0.032399 | 0.099045 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.40e-03 | 0.01640 | 0.01344 | 0.031056 | 0.1043 |

### K14 = K(p,alpha,a)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 8.31e-04 | 0.00966 | 0.00562 | 0.012903 | 0.13878 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.87e-04 | 0.00458 | 0.00278 | 0.0064191 | 0.079062 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.32e-04 | 0.00510 | 0.00316 | 0.0072948 | 0.075998 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.84e-04 | 0.00454 | 0.00300 | 0.0069109 | 0.047595 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 1.04e-04 | 0.00341 | 0.00172 | 0.0039608 | 0.043244 |

### K15 = K(p,MCN,a)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 3.68e-04 | 0.00643 | 0.00359 | 0.0082431 | 0.081873 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 1.10e-04 | 0.00351 | 0.00230 | 0.0053009 | 0.039958 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 1.10e-04 | 0.00351 | 0.00218 | 0.0050211 | 0.039209 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.77e-04 | 0.00445 | 0.00233 | 0.0053634 | 0.054841 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 3.60e-04 | 0.00635 | 0.00323 | 0.007443 | 0.067297 |

### K16 = K(p,alpha,MCN,a)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 7.61e-04 | 0.00924 | 0.00475 | 0.010893 | 0.1253 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.59e-05 | 0.00170 | 0.00069 | 0.0015884 | 0.030528 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.67e-05 | 0.00173 | 0.00071 | 0.0016421 | 0.030986 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 2.15e-05 | 0.00155 | 0.00063 | 0.0014569 | 0.029708 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.86e-05 | 0.00179 | 0.00064 | 0.0014793 | 0.036342 |

### K19 = K(p,MCN,q[kPa])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 1.30e-03 | 0.01206 | 0.00709 | 0.016248 | 0.12561 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 3.23e-05 | 0.00190 | 0.00101 | 0.0023197 | 0.038223 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 3.26e-05 | 0.00191 | 0.00096 | 0.0022143 | 0.040027 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 3.46e-05 | 0.00197 | 0.00089 | 0.0020575 | 0.044053 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 4.41e-05 | 0.00222 | 0.00091 | 0.0021005 | 0.051228 |

### K20 = K(p,alpha,MCN,q[kPa])

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 2.17e-03 | 0.01561 | 0.00995 | 0.022818 | 0.13431 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.27e-05 | 0.00160 | 0.00072 | 0.0016628 | 0.033634 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.17e-05 | 0.00156 | 0.00067 | 0.0015555 | 0.034423 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 1.78e-05 | 0.00141 | 0.00055 | 0.0012779 | 0.030145 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 2.21e-05 | 0.00157 | 0.00055 | 0.0012649 | 0.027184 |

### KN = K(p,alpha,r)

| RBF | kernel | degree | smoothing | neighbors | scaled | CV | 1 - R2 | RMSE | MAE | mean rel err | max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_1 | linear | 0 | 0.01 | None | no | RepeatedKFold(5x8) | 1.64e-03 | 0.01354 | 0.00878 | 0.020119 | 0.2027 |
| rbf_2 | thin_plate_spline | 1 | 0.001 | None | no | RepeatedKFold(5x8) | 2.55e-04 | 0.00535 | 0.00355 | 0.0081686 | 0.052624 |
| rbf_3 | thin_plate_spline | 1 | 1e-06 | 60 | no | RepeatedKFold(5x8) | 2.66e-04 | 0.00546 | 0.00360 | 0.0082894 | 0.056263 |
| rbf_4 | cubic | 1 | 1e-08 | 40 | no | RepeatedKFold(5x8) | 3.65e-04 | 0.00640 | 0.00412 | 0.0094832 | 0.074793 |
| rbf_5 | quintic | 2 | 1e-10 | 25 | no | RepeatedKFold(5x8) | 1.08e-03 | 0.01099 | 0.00631 | 0.01451 | 0.12399 |

## Methodological note

Improved cross-validation metrics from more complex RBF configurations indicate better predictive performance under this random-fold protocol. They do not by themselves imply a better physical explanation of `K`, especially because `log10(p)` dominates the raw variance and the predictors are correlated.
