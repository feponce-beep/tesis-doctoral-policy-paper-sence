# Firm-level training, wages and fiscal sensitivity under the SENCE Training Tax Credit (Franquicia Tributaria)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/feponce-beep/tesis-doctoral-policy-paper-sence/blob/main/LICENSE)
[![Data: ELE-4 to ELE-7](https://img.shields.io/badge/Data-ELE--4%20to%20ELE--7-blue)](https://www.economia.gob.cl)
[![Status: Working Paper](https://img.shields.io/badge/Status-Working%20Paper-orange)](https://github.com/feponce-beep/tesis-doctoral-policy-paper-sence)
[![Version](https://img.shields.io/badge/Version-v5.1-green)](https://github.com/feponce-beep/tesis-doctoral-policy-paper-sence)

---

## Overview

This repository contains the code, aggregated data outputs and documents for **Pilot Study**, part of a doctoral dissertation in Sociology at the **Universidad Complutense de Madrid (UCM)**, on workforce recertification and upskilling/reskilling strategies.

The study analyses the distribution of firm-level training activity, its association with wages and labour productivity, and the fiscal implications of eliminating the SENCE Training Tax Credit (Franquicia Tributaria SENCE), using microdata from the **Longitudinal Enterprise Survey (ELE-4 through ELE-7, reference years 2013-2022)**.

* Only **3.6%** of formal firms declare any collaborative training activity (**14.4%** large firms vs. **3.1%** SMEs vs. **2.9%** micro).
* The SME training-wage premium is stable at **~15%** (OLS, controlled for sector, age and group membership), averaging **15.2%** across four waves (2013: 15.1% · 2015: 11.3% · 2019: 19.4% · 2022: 15.0%).
* The concentration ratio (large/SME) increased from **2.43×** (2013) to **4.84×** (2022): the Matthew Effect in human capital.
* Fiscal break-even analysis shows the elimination of the FT is fiscally regressive if more than 44% of the observed wage differential is causally attributable to training — within the 40–70% range estimated by the international literature.

---

## Changes in v5.1

Two corrections relative to earlier versions:

1. **Physical headcount**: ELE variables `I151` (men) and `I160` (women) report person-months (Jan–Dec sum per INE questionnaire). Average annual headcount = `(I151 + I160) / 12`. Without this correction, derived variables (`PROD_LAB`, `SAL_TRAB`) were underestimated in absolute levels by a factor of 12, although OLS coefficients were unaffected.
2. **Numeric parsing**: `FE_TRANSVERSAL` and other numeric variables in ELE use comma as decimal separator (Chilean format). Without conversion, pandas reads them as strings and the expansion factor collapses to 1.

---

## Repository Structure

```
tesis-doctoral-policy-paper-sence/
|
|-- scripts/
|   |-- 01_analisis_principal.py            # ELE-7 full analysis: descriptives + OLS + figures 1-4
|   |-- 02_longitudinal_revalidacion.py     # Longitudinal revalidation ELE-4/5/6/7
|   |-- 03_figuras_longitudinales.py        # Longitudinal figures 5 and 6
|
|-- resultados/                             # Generated CSV outputs (reproducible)
|   |-- tabla1_concentracion_tamano.csv
|   |-- tabla2_brechas_sectoriales.csv
|   |-- tabla3_regresiones_OLS_v5.csv
|   |-- tabla4_medias_por_CAPALLY.csv
|   |-- tabla5_heterogeneidad_temporal.csv
|   |-- tabla_longitudinal_v5.csv
|
|-- figuras/                                # Generated PNG figures
|   |-- figura1_concentracion_tamano.png
|   |-- figura2_brechas_sectoriales.png
|   |-- figura3_prod_lab_violin.png
|   |-- figura4_prima_salarial_IC.png
|   |-- figura5_tasas_longitudinal.png
|   |-- figura6_prima_salarial_longitudinal.png
|
|-- datos/                                  # Input data (NOT included: confidential)
|
|-- README.md
|-- LICENSE
```

### Execution order

```bash
python scripts/01_analisis_principal.py          # Generates Figures 1-4 and Tables 1-5
python scripts/02_longitudinal_revalidacion.py   # Generates tabla_longitudinal_v5.csv
python scripts/03_figuras_longitudinales.py      # Generates Figures 5 and 6
```

---

## Data

| Source | Description | Access |
| --- | --- | --- |
| ELE-7 (Ministerio de Economía, 2024) | 7th Longitudinal Enterprise Survey, 2022 | Public: economia.gob.cl |
| ELE-4 to ELE-6 (INE / Ministerio de Economía) | Earlier waves for harmonised series (2013, 2015, 2019) | Public: ine.gob.cl |
| DIPRES (2025) | SENCE FT impact evaluation | Public report |
| OECD (2018, 2019) | Social mobility & adult learning benchmarks | Public |

> **Important:** Raw ELE microdata files are NOT included in this repository due to confidentiality restrictions. To replicate the analysis, download the official ELE CSV modules from economia.gob.cl and place them in `datos/` with the following filenames:
> - `ele7-full.csv`
> - `BBDD-ELE5-Formato-Texto.csv`
> - `BBDD_ELE4.csv`
> - (ELE-6 filename as referenced in `02_longitudinal_revalidacion.py`)

---

## Methods

* **Training proxy (`CAP_ALLY`)**: Binary indicator derived from the inter-firm collaboration motives block (ELE variables `D097`, `D176`, `D106`). Captures declared collaborative training activity — a conservative lower bound, not a direct measure of SENCE FT usage. In ELE-4 only `D097` and `D106` are available.
* **OLS models**: Log wage bill per worker and log sales per worker, controlling for sector (CIIU 2-digit fixed effects), firm age, group membership and size category. Robust HC3 standard errors (MacKinnon-White 1985), implemented in pure NumPy (no `statsmodels` dependency).
* **Harmonised series (ELE-4 to ELE-7)**: Best-available proxy for training in each wave, documented in `02_longitudinal_revalidacion.py`; results constitute a *harmonised series*, not a strict panel.
* **Fiscal sensitivity (break-even)**: Parametric simulation of net fiscal balance as a function of α (causal fraction of wage differential), including payroll taxes, health contributions and pension contributions (~20% of wage mass).

---

## Limitations

* `CAP_ALLY` is a proxy, not a direct measure of SENCE FT usage. The ELE contains no variable identifying FT users.
* OLS models are associative, not causal. Causal identification requires an administrative linkage (SENCE registry × ELE via RUT) under an institutional agreement not yet publicly available.
* The harmonised series uses non-identical variables across waves; all differences are documented in `02_longitudinal_revalidacion.py`.
* Fiscal simulation uses conservative default parameters; results should be updated with official DIPRES and SII figures.

---

## Citation

If you use this code or outputs, please cite:

```
Ponce Bollmann, F. (2026). Firm-level training, wages and fiscal sensitivity
under the SENCE Training Tax Credit. Pilot Study 2 (v5.1). Doctoral Research
in Sociology. Universidad Complutense de Madrid.
https://github.com/feponce-beep/tesis-doctoral-policy-paper-sence
```

---

## Author

**Felipe Ponce Bollmann**
Sociologist | Doctoral Researcher
Departamento de Sociología — Universidad Complutense de Madrid (UCM)

---

## License

Code: MIT License (see LICENSE file).
Documents (policy brief, column): Creative Commons Attribution 4.0 International (CC BY 4.0).
Data outputs: see data source licenses above.
