# =============================================================================
# fase3a_serie_longitudinal_ele.py
# Tesis Doctoral - Policy Paper | Franquicia Tributaria SENCE
# Universidad Complutense de Madrid - Doctorado en Sociologia
# =============================================================================
# Descripcion:
#   Fase 3a: Construccion y analisis de serie longitudinal de empresas ELE
#   Vinculacion de ondas ELE (2013, 2015, 2017, 2020, 2022) para identificar
#   efectos del cambio normativo sobre uso de Franquicia Tributaria SENCE.
#
# Estrategia de identificacion:
#   - Diferencias en diferencias (DiD) con empresas tratadas/control
#   - Ventana de evento: eliminacion beneficio FT (reforma 2020)
#   - Variable dependiente: log(salario_medio), tasa_uso_FT, hrs_capacitacion
#
# Datos: ELE panel simulado (sin microdatos confidenciales INE)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Configuracion
# ---------------------------------------------------------------------------
RAND_SEED = 42
np.random.seed(RAND_SEED)

OUTPUT_DIR = Path("outputs/fase3a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_TREAT   = "#003A70"   # azul UCM - grupo tratado
COLOR_CONTROL = "#C8102E"   # rojo UCM - grupo control
COLOR_REFORM  = "#FFA500"   # linea reforma

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 1. Construccion del panel longitudinal simulado
#    Basado en estructura ELE: panel de empresas con identificador comun
# ---------------------------------------------------------------------------

ONDAS = [2013, 2015, 2017, 2020, 2022]
N_EMPRESAS = 2_000  # empresas en panel balanceado (simulado)

np.random.seed(RAND_SEED)

# Caracteristicas fijas de empresa (time-invariant)
tamano = np.random.choice(
    ["Pequena", "Mediana", "Grande"],
    size=N_EMPRESAS,
    p=[0.45, 0.37, 0.18]
)
sector = np.random.choice(
    ["Comercio", "Industria", "Servicios", "Construccion"],
    size=N_EMPRESAS,
    p=[0.28, 0.22, 0.35, 0.15]
)

# Grupo tratado: empresas que usaban FT antes de reforma 2020
# Reforma: Ley 21.220 (2020) restringio acceso a Franquicia Tributaria
p_tratado = np.where(tamano == "Grande", 0.72,
             np.where(tamano == "Mediana", 0.45, 0.20))
tratado = np.array([np.random.binomial(1, p) for p in p_tratado])

# Construir panel
records = []
for t in ONDAS:
    # Tendencia base salarios (CLP, crecimiento ~3.5% anual)
    base_sal = {
        "Pequena": 520_000 * (1.035 ** (t - 2013)),
        "Mediana": 750_000 * (1.035 ** (t - 2013)),
        "Grande":  1_100_000 * (1.035 ** (t - 2013)),
    }

    # Efecto tratamiento pre-reforma: empresas tratadas tienen salarios ~8% mayores
    # Efecto post-reforma (2020+): reduccion ~6% para grupo tratado (efecto eliminacion FT)
    post_reforma = int(t >= 2020)

    for i in range(N_EMPRESAS):
        sal_base = base_sal[tamano[i]]
        efecto_trat = 0.08 * tratado[i]
        efecto_post = -0.06 * tratado[i] * post_reforma
        ruido = np.random.normal(0, 0.05)

        sal = sal_base * (1 + efecto_trat + efecto_post + ruido)

        # Horas capacitacion
        hrs_base = 15 * tratado[i] + 5
        hrs_post = -8 * tratado[i] * post_reforma
        hrs = max(0, hrs_base + hrs_post + np.random.normal(0, 3))

        # Uso FT (0/1)
        p_uso = 0.8 * tratado[i] * (1 - 0.65 * post_reforma) + 0.05 * (1 - tratado[i])
        usa_ft = np.random.binomial(1, min(max(p_uso, 0), 1))

        records.append({
            "id_empresa": i + 1,
            "anio":       t,
            "tamano":     tamano[i],
            "sector":     sector[i],
            "tratado":    tratado[i],
            "post":       post_reforma,
            "salario":    round(sal, 0),
            "hrs_cap":    round(hrs, 1),
            "usa_ft":     usa_ft,
        })

panel = pd.DataFrame(records)
panel["log_salario"] = np.log(panel["salario"])

print(f"Panel longitudinal: {len(panel):,} obs | {N_EMPRESAS:,} empresas | {len(ONDAS)} ondas")
print(panel.groupby(["anio", "tratado"])["usa_ft"].mean().unstack().round(3))

# ---------------------------------------------------------------------------
# 2. Diferencias en Diferencias (DiD)
# ---------------------------------------------------------------------------
# Especificacion: Y_it = alpha + beta*Tratado_i + gamma*Post_t
#                        + delta*(Tratado_i x Post_t) + epsilon_it
# delta = efecto causal de la reforma

# Subset 2017 (pre) vs 2022 (post) para DiD simple
df_did = panel[panel["anio"].isin([2017, 2022])].copy()
df_did["did"] = df_did["tratado"] * df_did["post"]

from numpy.linalg import lstsq

# DiD manual con regresion OLS
X = df_did[["tratado", "post", "did"]].copy()
X.insert(0, "const", 1)
y_sal = df_did["log_salario"]
y_hrs = df_did["hrs_cap"]
y_ft  = df_did["usa_ft"]

beta_sal, _, _, _ = lstsq(X.values, y_sal.values, rcond=None)
beta_hrs, _, _, _ = lstsq(X.values, y_hrs.values, rcond=None)
beta_ft,  _, _, _ = lstsq(X.values, y_ft.values,  rcond=None)

print("\n--- Estimacion DiD (2017 vs 2022) ---")
print(f"{'Variable':<20} {'alpha':>10} {'Tratado':>10} {'Post':>10} {'DiD (delta)':>12}")
print("-" * 65)
for label, beta in [("log(salario)", beta_sal), ("hrs_cap", beta_hrs), ("usa_ft", beta_ft)]:
    print(f"{label:<20} {beta[0]:>10.4f} {beta[1]:>10.4f} {beta[2]:>10.4f} {beta[3]:>12.4f}")

# ---------------------------------------------------------------------------
# 3. Event Study: tendencias paralelas
# ---------------------------------------------------------------------------
# Coeficientes por onda relativa a 2017 (anio base)
results_event = []
for t in ONDAS:
    sub = panel[panel["anio"] == t]
    media_trat  = sub[sub["tratado"] == 1]["log_salario"].mean()
    media_ctrl  = sub[sub["tratado"] == 0]["log_salario"].mean()
    se_trat = sub[sub["tratado"] == 1]["log_salario"].sem()
    se_ctrl = sub[sub["tratado"] == 0]["log_salario"].sem()
    results_event.append({
        "anio": t, "media_trat": media_trat, "media_ctrl": media_ctrl,
        "se_trat": se_trat, "se_ctrl": se_ctrl
    })

ev = pd.DataFrame(results_event)

# ---------------------------------------------------------------------------
# 4. Visualizaciones
# ---------------------------------------------------------------------------

# Fig 1: Event study - tendencias log(salario)
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(ev["anio"], ev["media_trat"], yerr=1.96*ev["se_trat"],
            color=COLOR_TREAT, marker="o", lw=2, label="Tratado (usaba FT)",
            capsize=4, capthick=1.5)
ax.errorbar(ev["anio"], ev["media_ctrl"], yerr=1.96*ev["se_ctrl"],
            color=COLOR_CONTROL, marker="s", lw=2, label="Control (no usaba FT)",
            capsize=4, capthick=1.5, linestyle="--")
ax.axvline(2020, color=COLOR_REFORM, linestyle=":", lw=2, label="Reforma FT (2020)")
ax.set_xlabel("Onda ELE", fontsize=11)
ax.set_ylabel("log(Salario medio)", fontsize=11)
ax.set_title("Event Study: log(Salario) por grupo tratado/control\n(Panel ELE simulado, 2013-2022)",
             fontsize=12, pad=12)
ax.set_xticks(ONDAS)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_event_study_salario.png")
plt.close()
print("Figura guardada: fig1_event_study_salario.png")

# Fig 2: Evolucion uso FT por grupo
usa_ft_mean = panel.groupby(["anio", "tratado"])["usa_ft"].mean().unstack()
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(usa_ft_mean.index, usa_ft_mean[1]*100, color=COLOR_TREAT,
        marker="o", lw=2, label="Tratado")
ax.plot(usa_ft_mean.index, usa_ft_mean[0]*100, color=COLOR_CONTROL,
        marker="s", lw=2, linestyle="--", label="Control")
ax.axvline(2020, color=COLOR_REFORM, linestyle=":", lw=2, label="Reforma 2020")
ax.set_xlabel("Onda ELE", fontsize=11)
ax.set_ylabel("% empresas que usan FT", fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
ax.set_title("Evolucion uso Franquicia Tributaria por grupo\n(Panel ELE simulado, 2013-2022)",
             fontsize=12, pad=12)
ax.set_xticks(ONDAS)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_evolucion_uso_ft.png")
plt.close()
print("Figura guardada: fig2_evolucion_uso_ft.png")

print("\n[Fase 3a completada] Outputs en:", OUTPUT_DIR)
