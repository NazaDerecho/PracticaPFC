import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analisis import datos, FS_EDA_PROCESADA, carpeta_resultados


# ============================================================
# 1. EXTRAER VARIABLES DE EDA
# ============================================================

resultados_eda = []


for sujeto in datos:

    if "EDA_procesada" not in datos[sujeto]:
        continue

    print(f"Extrayendo variables de {sujeto}")


    # --------------------------------------------------------
    # Obtener señal procesada
    # --------------------------------------------------------

    signals = datos[sujeto]["EDA_procesada"]


    # ========================================================
    # SCL - COMPONENTE TÓNICA
    # ========================================================

    scl_media = signals["EDA_Tonic"].mean()

    scl_max = signals["EDA_Tonic"].max()


    # ========================================================
    # SCR - COMPONENTE FÁSICA
    # ========================================================

    indices_scr = np.where(
        signals["SCR_Peaks"] == 1
    )[0]


    cantidad_scr = len(indices_scr)


    # ========================================================
    # AMPLITUD DE SCR
    # ========================================================

    amplitudes = (
        signals.loc[
            signals["SCR_Peaks"] == 1,
            "SCR_Amplitude"
        ]
        .dropna()
    )


    if len(amplitudes) > 0:

        scr_amplitud_media = amplitudes.mean()

        scr_amplitud_max = amplitudes.max()

    else:

        scr_amplitud_media = 0

        scr_amplitud_max = 0


    # ========================================================
    # COMPONENTE FÁSICA
    # ========================================================

    phasic_media = (
        signals["EDA_Phasic"].mean()
    )


    phasic_positiva = np.clip(
        signals["EDA_Phasic"].to_numpy(),
        0,
        None
    )


    phasic_area = np.trapezoid(
        phasic_positiva,
        dx=1 / FS_EDA_PROCESADA
    )


    # ========================================================
    # DURACIÓN
    # ========================================================

    duracion_min = (
        len(signals)
        / FS_EDA_PROCESADA
        / 60
    )


    # ========================================================
    # SCR POR MINUTO
    # ========================================================

    scr_por_min = (
        cantidad_scr / duracion_min
    )


    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================

    resultados_eda.append({

        "Sujeto": sujeto,

        "Duracion_min": duracion_min,

        "SCL_media": scl_media,

        "SCL_max": scl_max,

        "Cantidad_SCR": cantidad_scr,

        "SCR_por_min": scr_por_min,

        "SCR_amplitud_media":
            scr_amplitud_media,

        "SCR_amplitud_max":
            scr_amplitud_max,

        "Phasic_media":
            phasic_media,

        "Phasic_area":
            phasic_area
    })


# ============================================================
# 2. CREAR TABLA
# ============================================================

tabla_eda = pd.DataFrame(
    resultados_eda
)


print("\nTABLA DE VARIABLES EDA")

print(tabla_eda)

# ============================================================
# 3. ETIQUETAS DE ANSIEDAD

etiquetas_ansiedad = {
    "subject_01": "Ansiedad baja",
    "subject_02": "Ansiedad alta",
    "subject_03": "Ansiedad moderada"
}

tabla_eda["Ansiedad"] = (
    tabla_eda["Sujeto"]
    .map(etiquetas_ansiedad)
)

# ============================================================
# 4. ETIQUETAS DE CONDICIÓN
# ============================================================

etiquetas_condicion = {

    "baseline": "Sin ansiedad",
    "stress": "Ansiedad",
    "recovery": "Recuperación"
}

# ============================================================
# 5. GUARDAR TABLA

archivo_salida = (
    carpeta_resultados /
    "variables_EDA_ansiedad.csv"
)


tabla_eda.to_csv(
    archivo_salida,
    index=False
)


print("TABLA EDA + ANSIEDAD CREADA")

print(
    f"\nArchivo guardado en:\n"
    f"{archivo_salida}"
)

# ============================================================
# 6. HEATMAP COMPARATIVO DE VARIABLES EDA POR SUJETO
# ============================================================

# Variables que queremos comparar
variables_comparacion = [
    "SCL_media",
    "SCL_max",
    "Cantidad_SCR",
    "SCR_por_min",
    "SCR_amplitud_media",
    "SCR_amplitud_max",
    "Phasic_media",
    "Phasic_area"
]

# Copiar tabla
tabla_heatmap = tabla_eda[
    ["Sujeto"] + variables_comparacion
].copy()

# Poner sujeto como índice
tabla_heatmap = tabla_heatmap.set_index("Sujeto")

# ============================================================
# NORMALIZACIÓN Z-SCORE
# ============================================================
#
# Esto permite comparar variables con distintas escalas.
#
# Valor positivo  -> por encima del promedio
# Valor negativo  -> por debajo del promedio
# Valor cercano 0 -> cerca del promedio
# ============================================================

tabla_normalizada = (
    tabla_heatmap - tabla_heatmap.mean()
) / tabla_heatmap.std()


# ============================================================
# CREAR FIGURA
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 12)
)

imagen = ax.imshow(
    tabla_normalizada.values,
    aspect="auto"
)


# ============================================================
# ETIQUETAS DE EJES
# ============================================================

ax.set_xticks(
    np.arange(
        len(variables_comparacion)
    )
)

ax.set_xticklabels(
    [
        "SCL media",
        "SCL máxima",
        "Cantidad SCR",
        "SCR/min",
        "Amplitud media SCR",
        "Amplitud máxima SCR",
        "Fásica media",
        "Área fásica"
    ],
    rotation=45,
    ha="right"
)


ax.set_yticks(
    np.arange(
        len(tabla_normalizada.index)
    )
)

ax.set_yticklabels(
    tabla_normalizada.index
)


# ============================================================
# TÍTULO
# ============================================================

ax.set_title(
    "Comparación normalizada de características EDA por sujeto",
    fontsize=16,
    pad=20
)


# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    imagen,
    ax=ax
)

cbar.set_label(
    "Valor normalizado (z-score)"
)


# ============================================================
# AGREGAR LOS VALORES EN CADA CELDA
# ============================================================

for i in range(
    tabla_normalizada.shape[0]
):

    for j in range(
        tabla_normalizada.shape[1]
    ):

        valor = (
            tabla_normalizada.iloc[
                i,
                j
            ]
        )

        ax.text(
            j,
            i,
            f"{valor:.1f}",
            ha="center",
            va="center",
            fontsize=7
        )


# ============================================================
# AJUSTAR DISEÑO
# ============================================================

plt.tight_layout()


# ============================================================
# GUARDAR
# ============================================================

archivo_heatmap = (
    carpeta_resultados /
    "heatmap_EDA_por_sujeto.png"
)

plt.savefig(
    archivo_heatmap,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nHeatmap comparativo guardado en:"
)

print(
    archivo_heatmap
)