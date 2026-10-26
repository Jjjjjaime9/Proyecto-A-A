"""Utilidades del EDA: estilo gráfico común, estadísticos de asociación y gráficos.

Se separan del notebook para que este se lea como un informe y para que todas las
figuras compartan el mismo estilo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

RUTA_FIGURAS = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Paleta: azul para "no diabetes / sin evento", naranja para "diabetes / evento".
AZUL = "#2F6690"
NARANJA = "#D9822B"
GRIS = "#8A8A8A"
ROJO = "#B23A48"
PALETA_DIABETES = {0: AZUL, 1: NARANJA}
ETIQUETAS_DIABETES = {0: "Sin diabetes", 1: "Con diabetes"}
CMAP_SECUENCIAL = "Blues"
CMAP_DIVERGENTE = sns.diverging_palette(240, 20, as_cmap=True)


def configurar_estilo() -> None:
    """Estilo único para todas las figuras del proyecto."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titleweight": "bold",
            "figure.titlesize": 13,
            "legend.frameon": False,
        }
    )
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def rejilla_x(ax: plt.Axes) -> None:
    """Rejilla vertical en lugar de horizontal (para barras horizontales)."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda la figura en `reports/figures/<nombre>.png` para el informe y las slides."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIGURAS / f"{nombre}.png")


# ---------------------------------------------------------------------------
# Estadísticos
# ---------------------------------------------------------------------------


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013). 0 = independencia, 1 = asociación total."""
    tabla = pd.crosstab(x, y)
    if min(tabla.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_corr - 1, r_corr - 1))))


def matriz_cramers_v(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Matriz simétrica de V de Cramér entre variables categóricas."""
    m = pd.DataFrame(np.eye(len(columnas)), index=columnas, columns=columnas)
    for i, a in enumerate(columnas):
        for b in columnas[i + 1 :]:
            sub = df[[a, b]].dropna()
            m.loc[a, b] = m.loc[b, a] = cramers_v(sub[a], sub[b])
    return m


def test_chi2(x: pd.Series, y: pd.Series) -> dict:
    """Test χ² de independencia + V de Cramér."""
    sub = pd.concat([x, y], axis=1).dropna()
    tabla = pd.crosstab(sub.iloc[:, 0], sub.iloc[:, 1])
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    return {"chi2": chi2, "gl": gl, "p_valor": p, "v_cramer": cramers_v(sub.iloc[:, 0], sub.iloc[:, 1])}


def delta_cliff(a: pd.Series, b: pd.Series) -> float:
    """δ de Cliff a partir de la U de Mann-Whitney: P(a>b) - P(a<b). |δ|: <0,15 despreciable,
    <0,33 pequeño, <0,47 mediano, ≥0,47 grande (Romano et al., 2006)."""
    a, b = a.dropna(), b.dropna()
    u = stats.mannwhitneyu(a, b).statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def test_mann_whitney(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    res = stats.mannwhitneyu(a, b)
    return {"U": res.statistic, "p_valor": res.pvalue, "delta_cliff": delta_cliff(a, b)}


def test_kruskal(df: pd.DataFrame, grupo: str, valor: str) -> dict:
    """Kruskal-Wallis + ε² (tamaño del efecto: ~0,01 pequeño, ~0,08 mediano, ≥0,26 grande)."""
    sub = df[[grupo, valor]].dropna()
    muestras = [g[valor].to_numpy() for _, g in sub.groupby(grupo)]
    h, p = stats.kruskal(*muestras)
    return {"H": h, "p_valor": p, "epsilon2": h / (len(sub) - 1)}


def ic_proporcion(exitos: int, n: int) -> tuple[float, float]:
    """IC 95 % de Wilson para una proporción."""
    return proportion_confint(exitos, n, alpha=0.05, method="wilson")


def tabla_prevalencia(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """Proporción de target=1 por categoría de `col`, con IC 95 % de Wilson."""
    g = df.groupby(col, observed=True)[target].agg(["sum", "count"])
    g["prevalencia"] = g["sum"] / g["count"]
    ic = [ic_proporcion(s, n) for s, n in zip(g["sum"], g["count"])]
    g["ic_inf"], g["ic_sup"] = zip(*ic)
    return g.rename(columns={"sum": "casos", "count": "n"})


def tabla_media_ic(df: pd.DataFrame, col: str, valor: str) -> pd.DataFrame:
    """Media de `valor` por categoría de `col` con IC 95 % (t de Student)."""
    g = df.groupby(col, observed=True)[valor].agg(["mean", "std", "count"])
    margen = stats.t.ppf(0.975, g["count"] - 1) * g["std"] / np.sqrt(g["count"])
    g["ic_inf"], g["ic_sup"] = g["mean"] - margen, g["mean"] + margen
    return g.rename(columns={"mean": "media", "count": "n"})


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


def barras_prevalencia(
    df: pd.DataFrame,
    col: str,
    target: str,
    ax: plt.Axes,
    etiquetas: dict | None = None,
    titulo: str | None = None,
    color: str = NARANJA,
) -> None:
    """Barras de prevalencia del target por categoría, con IC 95 % y línea de referencia global."""
    t = tabla_prevalencia(df, col, target)
    x = [str(etiquetas.get(i, i)) if etiquetas else str(i) for i in t.index]
    err = np.vstack([t["prevalencia"] - t["ic_inf"], t["ic_sup"] - t["prevalencia"]])
    ax.bar(x, t["prevalencia"], yerr=err, color=color, alpha=0.85, capsize=3, edgecolor="white")
    ax.axhline(df[target].mean(), color=GRIS, ls="--", lw=1, label="Media global")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(titulo or col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45 if len(x) > 4 else 0)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right" if len(x) > 4 else "center")


def puntos_media_ic(
    df: pd.DataFrame,
    col: str,
    valor: str,
    ax: plt.Axes,
    etiquetas: dict | None = None,
    titulo: str | None = None,
    color: str = AZUL,
) -> None:
    """Media del valor por categoría con barras de error del IC 95 %."""
    t = tabla_media_ic(df, col, valor)
    x = [str(etiquetas.get(i, i)) if etiquetas else str(i) for i in t.index]
    err = np.vstack([t["media"] - t["ic_inf"], t["ic_sup"] - t["media"]])
    # Solo se unen los puntos si la variable es ordinal (más de dos categorías).
    fmt = "o-" if len(x) > 2 else "o"
    ax.errorbar(x, t["media"], yerr=err, fmt=fmt, color=color, capsize=3, lw=1.5, ms=6)
    if len(x) == 2:
        ax.set_xlim(-0.5, 1.5)
    ax.axhline(df[valor].mean(), color=GRIS, ls="--", lw=1)
    ax.set_title(titulo or col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45 if len(x) > 4 else 0)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right" if len(x) > 4 else "center")


def heatmap_correlacion(
    matriz: pd.DataFrame, ax: plt.Axes, titulo: str, vmin: float = -1, cmap=None
) -> None:
    """Heatmap triangular inferior con anotaciones."""
    mascara = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    sns.heatmap(
        matriz,
        mask=mascara,
        ax=ax,
        cmap=cmap or CMAP_DIVERGENTE,
        vmin=vmin,
        vmax=1,
        center=0 if vmin < 0 else None,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        square=True,
    )
    ax.grid(False)
    ax.set_title(titulo)
