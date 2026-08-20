"""
Paleta del dashboard IRF-N -- tema oscuro. Fuente unica de verdad del color de
la interfaz (R9: presentacion, jamas modelo -- aqui solo se da color a valores
que el artefacto ya calculo).

Regla de asignacion:
  - Indice IRF-N (0->1, sin signo)      -> IRFN_SCALE (secuencial, viridis truncado)
  - Retorno / velas (con signo)         -> DIRECTIONAL
  - Series calculadas (MA, MACD, ...)   -> SERIES
  - Correlaciones / z-scores            -> DIVERGING_DARK
El verde y el rojo quedan reservados exclusivamente para el signo del precio.

Uso tipico en una pantalla:
    from theme import apply_layout, REGIME_COLORS, IRFN_SCALE
    fig = go.Figure(...)
    apply_layout(fig)                       # chrome oscuro
    st.plotly_chart(fig, theme=None)        # theme=None: respeta ESTE chrome
"""
from __future__ import annotations

# ── Chrome del dashboard ──────────────────────────────────────────────
CHROME = {
    "bg":        "#0E1117",  # lienzo (default dark de Streamlit)
    "surface":   "#161A23",  # tarjetas, sidebar, paneles
    "grid":      "#1C222E",  # lineas de grilla
    "border":    "#262C38",  # separadores
    "text":      "#D7DCE5",  # texto principal
    "text_dim":  "#8B93A5",  # etiquetas de eje, texto secundario
    "accent_ui": "#5B8DEF",  # botones, sliders, foco (NO codifica dato)
    "no_data":   "#3A4150",  # NaN, gaps, ventanas sin observacion
}

# ── Indice IRF-N: viridis truncado en 0.15 para contraste sobre #0E1117 ──
IRFN_SCALE = [
    (0.00, "#443A83"),
    (0.14, "#3E4A89"),
    (0.28, "#31688E"),
    (0.42, "#26828E"),
    (0.56, "#1F9E89"),
    (0.70, "#35B779"),
    (0.82, "#6ECE58"),
    (0.92, "#B5DE2B"),
    (1.00, "#FDE725"),
]

# ── Bandas semanticas del indice (KPI, badges, sombreado de fondo) ────
IRFN_BANDS = [
    {"label": "Base",     "lo": 0.00, "hi": 0.35, "color": "#31688E"},
    {"label": "Elevado",  "lo": 0.35, "hi": 0.60, "color": "#1F9E89"},
    {"label": "Alto",     "lo": 0.60, "hi": 0.80, "color": "#B5DE2B"},
    {"label": "Extremo",  "lo": 0.80, "hi": 1.01, "color": "#FDE725"},
]

# ── Signo del precio (reservado) ─────────────────────────────────────
DIRECTIONAL = {"up": "#26A69A", "down": "#EF5350", "flat": "#787B86"}

# ── Series calculadas, hasta 4 activas por panel ─────────────────────
SERIES = ["#5B8DEF", "#FF8A3D", "#A47BE0", "#4FC3C7"]

# ── Divergente sobre fondo oscuro (correlaciones, z-score del indice) ──
DIVERGING_DARK = [
    (0.00, "#C9553D"),
    (0.25, "#8A4A45"),
    (0.50, "#3A4150"),
    (0.75, "#3F7FA8"),
    (1.00, "#5FB4E8"),
]

# ── Linea de referencia (media historica, umbral de disparo) ─────────
REFERENCE = {"color": "#8B93A5", "dash": "dash", "width": 1.2}


def band_for(value: float) -> dict:
    """Devuelve la banda semantica que corresponde a un valor del indice."""
    for band in IRFN_BANDS:
        if band["lo"] <= value < band["hi"]:
            return band
    return IRFN_BANDS[-1]


def plotly_scale(stops=IRFN_SCALE) -> list:
    """Convierte las tuplas a la forma que espera Plotly."""
    return [[pos, hexcode] for pos, hexcode in stops]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c)) for c in rgb))


def scale_color(pos: float, stops=IRFN_SCALE) -> str:
    """Interpola un color de IRFN_SCALE en la posicion pos in [0,1]."""
    pos = min(1.0, max(0.0, float(pos)))
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if p0 <= pos <= p1:
            t = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return _rgb_to_hex(tuple(a[i] + t * (b[i] - a[i]) for i in range(3)))
    return stops[-1][1]


def regime_colors(k: int) -> list:
    """Colores de K regimenes muestreados de IRFN_SCALE, de menor a mayor varianza
    (indice 0 = menor varianza = extremo BAJO del indice IRF-N; ultimo = extremo
    ALTO). Asi el color del regimen y la escala del indice hablan el mismo idioma
    visual, y verde/rojo quedan libres para el signo del precio."""
    if k <= 1:
        return [scale_color(0.30)]
    return [scale_color(i / (k - 1)) for i in range(k)]


# Lista por defecto que consumen las pantallas via components.REGIME_COLORS
# (indexada por el regimen k del modelo vigente). Para el caso real -- SPY K=2 y
# BTC K=1 -- da: 0 = baja volatilidad (teal, indice bajo), 1 = alta volatilidad
# (amarillo, indice extremo). Las entradas 2..4 cubren K mayores hipoteticos.
REGIME_COLORS = ["#31688E", "#FDE725", "#1F9E89", "#B5DE2B", "#6ECE58"]


def apply_layout(fig):
    """Aplica el chrome oscuro a una figura de Plotly."""
    fig.update_layout(
        paper_bgcolor=CHROME["bg"],
        plot_bgcolor=CHROME["bg"],
        colorway=SERIES,
        font=dict(color=CHROME["text"], family="IBM Plex Mono, monospace", size=12),
        xaxis=dict(gridcolor=CHROME["grid"], zerolinecolor=CHROME["border"],
                   linecolor=CHROME["border"], tickfont=dict(color=CHROME["text_dim"])),
        yaxis=dict(gridcolor=CHROME["grid"], zerolinecolor=CHROME["border"],
                   linecolor=CHROME["border"], tickfont=dict(color=CHROME["text_dim"])),
        margin=dict(l=48, r=20, t=36, b=36),
    )
    return fig


def _register_template() -> None:
    """Registra 'irfn_dark' como template por defecto de Plotly, para que las
    figuras adopten el chrome aunque una pantalla no llame apply_layout."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=CHROME["bg"],
        plot_bgcolor=CHROME["bg"],
        colorway=SERIES,
        font=dict(color=CHROME["text"], family="IBM Plex Mono, monospace", size=12),
        xaxis=dict(gridcolor=CHROME["grid"], zerolinecolor=CHROME["border"],
                   linecolor=CHROME["border"], tickfont=dict(color=CHROME["text_dim"])),
        yaxis=dict(gridcolor=CHROME["grid"], zerolinecolor=CHROME["border"],
                   linecolor=CHROME["border"], tickfont=dict(color=CHROME["text_dim"])),
    )
    pio.templates["irfn_dark"] = tpl
    pio.templates.default = "irfn_dark"


_register_template()
