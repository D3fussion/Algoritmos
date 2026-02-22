import random
import itertools
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


def inicializar_centroides(puntos, k):
    centroides = [random.choice(puntos)]

    while len(centroides) < k:
        distancias_cuadradas = []

        for punto in puntos:
            dist_a_centroides = [
                sum((punto[dim] - c[dim]) ** 2 for dim in range(len(punto)))
                for c in centroides
            ]
            distancias_cuadradas.append(min(dist_a_centroides))

        suma_distancias = sum(distancias_cuadradas)

        if suma_distancias == 0:
            centroides.append(random.choice(puntos))
            continue

        valor_aleatorio = random.uniform(0, suma_distancias)
        sumas_acumuladas = list(itertools.accumulate(distancias_cuadradas))

        for i, punto in enumerate(puntos):
            if sumas_acumuladas[i] >= valor_aleatorio:
                if punto not in centroides:
                    centroides.append(punto)
                break

    return [c[:] for c in centroides]


def dividir_lista_en_pares(lista):
    if not lista:
        return [], []

    num_elementos = len(lista[0])
    indices = list(itertools.combinations(range(num_elementos), 2))

    pares = [[[fila[i], fila[j]] for fila in lista] for i, j in indices]

    return pares, indices


def dividir_en_valor(lista):
    diccionario_tabla = dict()
    pares_ordenados = sorted((valor, indice) for indice, valor in enumerate(lista))

    for valor, grupo in itertools.groupby(pares_ordenados, key=lambda x: x[0]):
        diccionario_tabla[valor] = [item[1] for item in grupo]

    return diccionario_tabla


def sacar_distancias(puntos, centroides):
    tabla_de_puntos = []
    for punto in puntos:
        indice_mas_cercano = min(
            range(len(centroides)),
            key=lambda i: (
                (centroides[i][0] - punto[0]) ** 2 + (centroides[i][1] - punto[1]) ** 2
            ),
        )
        tabla_de_puntos.append(indice_mas_cercano)
    return tabla_de_puntos


def calcular_centros(puntos, diccionario_tabla, k):
    centroides = {}
    for key, value in diccionario_tabla.items():
        sumatoria_x = 0
        sumatoria_y = 0
        for i in value:
            sumatoria_x += puntos[i][0]
            sumatoria_y += puntos[i][1]

        centroide_x = sumatoria_x / len(value)
        centroide_y = sumatoria_y / len(value)
        centroides[key] = [centroide_x, centroide_y]

    if len(centroides) < k:
        for i in range(k):
            if i not in centroides:
                centroides[i] = list(random.choice(puntos))

    return centroides


def evaluar_rendimiento(asignaciones, clases_reales):
    clases_unicas = sorted(list(set(clases_reales)))
    clusters_encontrados = sorted(list(set(asignaciones)))

    mejor_mapeo = {}
    max_coincidencias = -1

    if len(clusters_encontrados) < len(clases_unicas):
        faltantes = set(range(len(clases_unicas))) - set(clusters_encontrados)
        clusters_encontrados.extend(list(faltantes))

    for permutacion in itertools.permutations(clases_unicas):
        mapeo = dict(zip(clusters_encontrados, permutacion))
        coincidencias = sum(
            1
            for asignado, real in zip(asignaciones, clases_reales)
            if mapeo.get(asignado) == real
        )
        if coincidencias > max_coincidencias:
            max_coincidencias = coincidencias
            mejor_mapeo = mapeo

    asignaciones_traducidas = [mejor_mapeo.get(a, -1) for a in asignaciones]

    metricas_por_clase = {}
    for clase in clases_unicas:
        TP = sum(
            1
            for a, r in zip(asignaciones_traducidas, clases_reales)
            if a == clase and r == clase
        )
        FP = sum(
            1
            for a, r in zip(asignaciones_traducidas, clases_reales)
            if a == clase and r != clase
        )
        FN = sum(
            1
            for a, r in zip(asignaciones_traducidas, clases_reales)
            if a != clase and r == clase
        )

        precision_clase = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_clase = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_clase = (
            2 * (precision_clase * recall_clase) / (precision_clase + recall_clase)
            if (precision_clase + recall_clase) > 0
            else 0
        )

        metricas_por_clase[clase] = {
            "Precision": precision_clase,
            "Recall": recall_clase,
            "F1-Score": f1_clase,
        }

    n_clases = len(clases_unicas)
    macro_precision = (
        sum(m["Precision"] for m in metricas_por_clase.values()) / n_clases
    )
    macro_recall = sum(m["Recall"] for m in metricas_por_clase.values()) / n_clases
    macro_f1 = sum(m["F1-Score"] for m in metricas_por_clase.values()) / n_clases
    aciertos = max_coincidencias / len(clases_reales)

    return aciertos, macro_precision, macro_recall, macro_f1, asignaciones_traducidas


def convertir_a_tabla(puntos_plot, asignaciones_reales, asignaciones_predichas):
    datos_raw = []
    for i, punto in enumerate(puntos_plot):
        datos_raw.append(
            {
                "features": punto,
                "class": asignaciones_reales[i],
                "prediccion": asignaciones_predichas[i],
            }
        )
    return datos_raw


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload-kmeans", methods=["POST"])
def subir_archivo():
    try:
        if "archivo" not in request.files:
            return jsonify({"error": "No se proporciono un archivo"}), 400

        archivo = request.files["archivo"]
        max_veces_raw = request.form.get("max_veces", "auto")

        if max_veces_raw.strip() == "" or max_veces_raw.strip().lower() == "auto":
            max_veces = 999
        else:
            try:
                max_veces = int(max_veces_raw)
            except ValueError:
                max_veces = 999

        contenido = archivo.read().decode("utf-8").strip().split("\n")

        puntos_plot = []
        clases = []

        for linea in contenido:
            if not linea.strip():
                continue

            partes = linea.split(",")
            if len(partes) < 3:
                return jsonify(
                    {
                        "error": "Se requieren al menos 3 columnas (ej. Feature1, Feature2, Clase)"
                    }
                ), 400

            puntos_plot.append([float(x) for x in partes[:-1]])
            clases.append(partes[-1].strip())

        k = len(set(clases))

        puntos_plot_pares, indices_pares = dividir_lista_en_pares(puntos_plot)
        resultados_analisis = []

        for index_par, puntos_plot_par in enumerate(puntos_plot_pares):
            cols_comparadas = indices_pares[index_par]

            clusters = inicializar_centroides(puntos_plot_par, k)
            lista_asignaciones = []

            for veces in itertools.count(1):
                lista_asignaciones = sacar_distancias(puntos_plot_par, clusters)
                lista_dividida = dividir_en_valor(lista_asignaciones)
                centros_dict = calcular_centros(puntos_plot_par, lista_dividida, k)

                clusters_new = [centros_dict[i] for i in sorted(centros_dict.keys())]

                if veces >= max_veces:
                    break
                elif clusters_new != clusters:
                    clusters = clusters_new
                else:
                    break

            aciertos, precision, recall, f1, asignaciones_traducidas = (
                evaluar_rendimiento(lista_asignaciones, clases)
            )

            resultados_analisis.append(
                {
                    "indices_features": cols_comparadas,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "aciertos": aciertos,
                    "centroides": clusters,
                    "asignaciones": lista_asignaciones,
                    "asignaciones_traducidas": asignaciones_traducidas,
                    "iteraciones": veces,
                }
            )

        resultados_analisis.sort(key=lambda x: x["f1"], reverse=True)

        mejor_resultado = resultados_analisis[0]
        datos_tabla = convertir_a_tabla(
            puntos_plot, clases, mejor_resultado["asignaciones_traducidas"]
        )

        return jsonify(
            {
                "numero_muestras": len(puntos_plot),
                "resultados": resultados_analisis,
                "datos_tabla": datos_tabla,
                "mejores_caracteristicas": mejor_resultado["indices_features"],
            }
        )

    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
