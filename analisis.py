import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from scipy.signal import resample_poly


# ============================================================
# 1. DEFINIR CARPETA PRINCIPAL
# ============================================================

carpeta_principal = Path(
    r"C:\Users\lourd\OneDrive\Escritorio\Naza\PFC\Subjects"
)


# ============================================================
# 2. DEFINIR SEÑALES
# ============================================================

senales = [
    "ACC",
    "BVP",
    "EDA",
    "HR",
    "IBI",
    "TEMP"
]


# ============================================================
# 3. BUSCAR AUTOMÁTICAMENTE TODAS LAS CARPETAS DE SUJETOS
# ============================================================

carpetas_sujetos = sorted([
    carpeta
    for carpeta in carpeta_principal.iterdir()
    if carpeta.is_dir()
])

print(
    "Cantidad de sujetos encontrados:",
    len(carpetas_sujetos)
)


# ============================================================
# 4. CARGAR TODOS LOS CSV
# ============================================================

datos = {}

for carpeta_sujeto in carpetas_sujetos:

    nombre_sujeto = carpeta_sujeto.name

    print(
        f"\nCargando: {nombre_sujeto}"
    )

    datos[nombre_sujeto] = {}

    for senal in senales:

        archivo = (
            carpeta_sujeto /
            f"{senal}.csv"
        )

        if archivo.exists():

            df = pd.read_csv(
                archivo,
                header=None
            )

            datos[nombre_sujeto][senal] = df

            print(
                f"   {senal}: "
                f"{df.shape[0]} filas x "
                f"{df.shape[1]} columnas"
            )

        else:

            print(
                f"   {senal}: "
                f"ARCHIVO NO ENCONTRADO"
            )


# ============================================================
# 5. RENOMBRAR COLUMNAS
# ============================================================

for sujeto in datos:

    if "ACC" in datos[sujeto]:

        datos[sujeto]["ACC"].columns = [
            "ACC_X",
            "ACC_Y",
            "ACC_Z"
        ]

    if "BVP" in datos[sujeto]:

        datos[sujeto]["BVP"].columns = [
            "BVP"
        ]

    if "EDA" in datos[sujeto]:

        datos[sujeto]["EDA"].columns = [
            "EDA"
        ]

    if "HR" in datos[sujeto]:

        datos[sujeto]["HR"].columns = [
            "HR"
        ]

    if "IBI" in datos[sujeto]:

        datos[sujeto]["IBI"].columns = [
            "Tiempo_IBI",
            "IBI"
        ]

    if "TEMP" in datos[sujeto]:

        datos[sujeto]["TEMP"].columns = [
            "TEMP"
        ]


# ============================================================
# 6. FRECUENCIAS DE MUESTREO
# ============================================================

FS_EDA = 4
FS_EDA_PROCESADA = 64

FS_ACC = 32
FS_BVP = 64


# ============================================================
# 7. CREAR EJES TEMPORALES
# ============================================================

for sujeto in datos:

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    if "EDA" in datos[sujeto]:

        eda = datos[sujeto]["EDA"]

        eda["Tiempo_s"] = (
            np.arange(len(eda))
            / FS_EDA
        )

        eda["Tiempo_min"] = (
            eda["Tiempo_s"]
            / 60
        )


    # --------------------------------------------------------
    # ACC
    # --------------------------------------------------------

    if "ACC" in datos[sujeto]:

        acc = datos[sujeto]["ACC"]

        acc["Tiempo_s"] = (
            np.arange(len(acc))
            / FS_ACC
        )

        acc["Tiempo_min"] = (
            acc["Tiempo_s"]
            / 60
        )


    # --------------------------------------------------------
    # BVP
    # --------------------------------------------------------

    if "BVP" in datos[sujeto]:

        bvp = datos[sujeto]["BVP"]

        bvp["Tiempo_s"] = (
            np.arange(len(bvp))
            / FS_BVP
        )

        bvp["Tiempo_min"] = (
            bvp["Tiempo_s"]
            / 60
        )


# ============================================================
# 8. CREAR CARPETAS PARA RESULTADOS
# ============================================================

carpeta_resultados = (
    carpeta_principal.parent /
    "resultados"
)


# Mantengo la misma carpeta que ya venías usando

carpeta_eda = (
    carpeta_resultados /
    "EDA_cruda"
)

carpeta_bvp = (
    carpeta_resultados /
    "BVP_cruda"
)

carpeta_acc = (
    carpeta_resultados /
    "ACC_cruda"
)


carpeta_eda.mkdir(
    parents=True,
    exist_ok=True
)

carpeta_bvp.mkdir(
    parents=True,
    exist_ok=True
)

carpeta_acc.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. PROCESAR EDA CON NEUROKIT2
# ============================================================

for sujeto in datos:

    if "EDA" not in datos[sujeto]:
        continue

    print("\n========================================")
    print(f"Procesando EDA: {sujeto}")
    print("========================================")

    try:

        # ====================================================
        # 9.1 OBTENER EDA CRUDA
        # ====================================================

        eda_df = datos[sujeto]["EDA"]

        eda_raw = (
            pd.to_numeric(
                eda_df["EDA"],
                errors="coerce"
            )
            .to_numpy(dtype=float)
        )


        print(
            "Cantidad de muestras originales:",
            len(eda_raw)
        )


        # ====================================================
        # 9.2 CONTROLAR NaN E INFINITOS
        # ====================================================

        # Convertir infinitos en NaN
        eda_raw[
            ~np.isfinite(eda_raw)
        ] = np.nan


        cantidad_nan = np.isnan(
            eda_raw
        ).sum()

        print(
            "NaN / infinitos encontrados:",
            cantidad_nan
        )


        # Interpolar si existen NaN
        if cantidad_nan > 0:

            eda_raw = (
                pd.Series(eda_raw)
                .interpolate(
                    method="linear",
                    limit_direction="both"
                )
                .to_numpy()
            )


        # ====================================================
        # 9.3 VERIFICAR QUE LA SEÑAL NO SEA CONSTANTE
        # ====================================================

        desviacion = np.std(
            eda_raw
        )

        print(
            f"Desvío estándar EDA: "
            f"{desviacion:.6f}"
        )


        if desviacion == 0:

            print(
                f"ATENCIÓN: {sujeto} tiene "
                f"una señal EDA constante."
            )

            continue


        # ====================================================
        # 9.4 INFORMACIÓN BÁSICA
        # ====================================================

        print(
            f"EDA mínima: "
            f"{np.min(eda_raw):.4f} µS"
        )

        print(
            f"EDA máxima: "
            f"{np.max(eda_raw):.4f} µS"
        )


        # ====================================================
        # 9.5 TIEMPO ORIGINAL 4 Hz
        # ====================================================

        tiempo_raw_min = (
            np.arange(
                len(eda_raw)
            )
            / FS_EDA
            / 60
        )


        # ====================================================
        # 9.6 UPSAMPLING 4 Hz -> 64 Hz
        # ====================================================

        eda_64hz = resample_poly(
            eda_raw,
            up=16,
            down=1
        )


        print(
            "Muestras luego del upsampling:",
            len(eda_64hz)
        )


        # ====================================================
        # 9.7 PROCESAMIENTO NEUROKIT2
        # ====================================================

        signals, info = nk.eda_process(
            eda_64hz,
            sampling_rate=64
        )


        print(
            "NeuroKit2 procesó correctamente."
        )


        # ====================================================
        # 9.8 EJE TEMPORAL PROCESADO
        # ====================================================

        tiempo_procesado_min = (
            np.arange(
                len(signals)
            )
            / 64
            / 60
        )


        # ====================================================
        # 9.9 GUARDAR RESULTADO
        # ====================================================

        datos[sujeto][
            "EDA_procesada"
        ] = signals.copy()

        datos[sujeto][
            "EDA_procesada"
        ]["Tiempo_min"] = (
            tiempo_procesado_min
        )


        # ====================================================
        # 9.10 DETECTAR SCR
        # ====================================================

        indices_picos = np.where(
            signals["SCR_Peaks"]
            == 1
        )[0]


        numero_scr = len(
            indices_picos
        )


        print(
            f"SCR detectadas: "
            f"{numero_scr}"
        )


        # ====================================================
        # 9.11 CREAR FIGURA
        # ====================================================

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(15, 12)
        )


        # ----------------------------------------------------
        # GRÁFICO 1: EDA CRUDA
        # ----------------------------------------------------

        axes[0].plot(
            tiempo_raw_min,
            eda_raw
        )

        axes[0].set_title(
            "1. EDA cruda original (4 Hz)"
        )

        axes[0].set_ylabel(
            "EDA [µS]"
        )

        axes[0].grid(True)


        # ----------------------------------------------------
        # GRÁFICO 2: EDA LIMPIA
        # ----------------------------------------------------

        axes[1].plot(
            tiempo_procesado_min,
            signals["EDA_Clean"]
        )

        axes[1].set_title(
            "2. EDA limpia / filtrada"
        )

        axes[1].set_ylabel(
            "EDA [µS]"
        )

        axes[1].grid(True)


        # ----------------------------------------------------
        # GRÁFICO 3: COMPONENTE TÓNICA
        # ----------------------------------------------------

        axes[2].plot(
            tiempo_procesado_min,
            signals["EDA_Tonic"]
        )

        axes[2].set_title(
            "3. Componente tónica (SCL)"
        )

        axes[2].set_ylabel(
            "SCL [µS]"
        )

        axes[2].grid(True)


        # ----------------------------------------------------
        # GRÁFICO 4: COMPONENTE FÁSICA
        # ----------------------------------------------------

        axes[3].plot(
            tiempo_procesado_min,
            signals["EDA_Phasic"],
            label="Componente fásica"
        )


        if numero_scr > 0:

            axes[3].scatter(

                tiempo_procesado_min[
                    indices_picos
                ],

                signals[
                    "EDA_Phasic"
                ].iloc[
                    indices_picos
                ],

                marker="o",

                label=(
                    f"SCR detectadas "
                    f"(n={numero_scr})"
                )
            )


        axes[3].set_title(
            "4. Componente fásica (SCR) "
            "y respuestas detectadas"
        )

        axes[3].set_xlabel(
            "Tiempo [min]"
        )

        axes[3].set_ylabel(
            "SCR [µS]"
        )

        axes[3].grid(True)

        axes[3].legend()


        # ====================================================
        # 9.12 TÍTULO GENERAL
        # ====================================================

        fig.suptitle(
            f"Procesamiento de EDA - {sujeto}",
            fontsize=16
        )

        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.97
            ]
        )


        # ====================================================
        # 9.13 GUARDAR IMAGEN
        # ====================================================

        nombre_archivo = (
            carpeta_eda /
            f"{sujeto}_EDA_procesada.png"
        )


        plt.savefig(
            nombre_archivo,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            f"Imagen guardada correctamente:"
        )

        print(
            nombre_archivo
        )


    # ========================================================
    # SI UN SUJETO PRODUCE ERROR
    # ========================================================

    except Exception as error:

        plt.close("all")

        print(
            "\n******** ERROR ********"
        )

        print(
            f"Error procesando: {sujeto}"
        )

        print(
            f"Tipo de error: "
            f"{type(error).__name__}"
        )

        print(
            f"Mensaje: {error}"
        )

        print(
            "Se continúa con el siguiente sujeto."
        )

        print(
            "************************\n"
        )


print(
    "\nProcesamiento EDA finalizado."
)