import random
import itertools
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


def inicializar_centroides(puntos, k):
    """
    Inicializa los centroides usando el método K-Means++ (El otro método era 
    con la función Húngara, pero era demasiado costosa computacionalmente y 
    difícil de implementar).
    El primer centroide se elige al azar, y los siguientes se eligen
    con probabilidad proporcional a su distancia al centroide más cercano,
    para lograr una mejor distribución inicial.
    """
    # Elige el primer centroide aleatoriamente
    centroides = [random.choice(puntos)]

    while len(centroides) < k:
        distancias_cuadradas = []

        # Calcula la distancia cuadrada de cada punto al centroide más cercano (Sin raiz, ya que no es necesario para la comparación)
        for punto in puntos:
            dist_a_centroides = [
                sum((punto[dim] - c[dim]) ** 2 for dim in range(len(punto)))
                for c in centroides
            ]
            distancias_cuadradas.append(min(dist_a_centroides))

        suma_distancias = sum(distancias_cuadradas)

        # Si todos los puntos coinciden con algún centroide, elige uno al azar
        if suma_distancias == 0:
            centroides.append(random.choice(puntos))
            continue

        # Selecciona el siguiente centroide con probabilidad proporcional a la distancia
        valor_aleatorio = random.uniform(0, suma_distancias)
        sumas_acumuladas = list(itertools.accumulate(distancias_cuadradas))

        for i, punto in enumerate(puntos):
            if sumas_acumuladas[i] >= valor_aleatorio:
                if punto not in centroides:
                    centroides.append(punto)
                break

    # Devuelve copias de los centroides para evitar mutaciones externas
    return [c[:] for c in centroides]


def dividir_lista_en_pares(lista):
    """
    Genera todas las combinaciones posibles de 2 features (columnas)
    de la lista de puntos, lo cual es útil para analizar cada par de 
    features por separado.

    Return:
        1. pares: lista de sub-datasets con solo 2 dimensiones cada uno.
        2. indices: [()]: los índices originales de las columnas usadas en cada par.
    """
    if not lista:
        return [], []

    num_elementos = len(lista[0])
    # Obtiene todas las combinaciones de 2 índices entre las columnas disponibles
    indices = list(itertools.combinations(range(num_elementos), 2))

    # Guarda cada punto del dataset a 2 dimensiones
    pares = [[[fila[i], fila[j]] for fila in lista] for i, j in indices]

    return pares, indices


def dividir_en_valor(lista):
    """
    Agrupa los índices de los puntos según su cluster asignado.

    Ejemplo: [0, 1, 0, 2] → {0: [0, 2], 1: [1], 2: [3]}

    Return: 
        1. diccionario_tabla: Un diccionario donde la clave es el ID 
           del cluster y el valor es la lista de índices de puntos que le pertenecen.
    """
    diccionario_tabla = dict()
    # Ordena los pares (valor de cluster, índice del punto)
    pares_ordenados = sorted((valor, indice) for indice, valor in enumerate(lista))

    # Agrupa por cluster y guarda los índices correspondientes
    for valor, grupo in itertools.groupby(pares_ordenados, key=lambda x: x[0]):
        diccionario_tabla[valor] = [item[1] for item in grupo]

    return diccionario_tabla


def sacar_distancias(puntos, centroides):
    """
    Asigna cada punto al centroide más cercano según la distancia euclídea al cuadrado.

    Return: 
        1. tabla_de_puntos: una lista con el índice del centroide más cercano para cada punto.
    """
    tabla_de_puntos = []
    for punto in puntos:
        # Encuentra el índice del centroide con menor distancia al punto actual
        indice_mas_cercano = min(
            range(len(centroides)),
            key=lambda i: (
                (centroides[i][0] - punto[0]) ** 2 + (centroides[i][1] - punto[1]) ** 2
            ),
        )
        tabla_de_puntos.append(indice_mas_cercano)
    return tabla_de_puntos


def calcular_centros(puntos, diccionario_tabla, k):
    """
    Recalcula la posición de cada centroide como el promedio de los puntos
    que pertenecen a su cluster.

    Si algún cluster quedó vacío (sin puntos asignados), se le asigna
    un punto al azar para evitar clusters muertos.

    Return: 
        1. centroides: un diccionario {id_cluster: [x, y]}.
    """
    centroides = {}
    for key, value in diccionario_tabla.items():
        sumatoria_x = 0
        sumatoria_y = 0

        # Suma las coordenadas de todos los puntos del cluster
        for i in value:
            sumatoria_x += puntos[i][0]
            sumatoria_y += puntos[i][1]

        # Calcula el promedio para obtener el nuevo centroide
        centroide_x = sumatoria_x / len(value)
        centroide_y = sumatoria_y / len(value)
        centroides[key] = [centroide_x, centroide_y]

    # Si hay menos centroides que k, reasigna un punto aleatorio a los clusters vacíos
    # La idea con esto es evitar que algún cluster quede vacío
    if len(centroides) < k:
        for i in range(k):
            if i not in centroides:
                centroides[i] = list(random.choice(puntos))

    return centroides


def evaluar_rendimiento(asignaciones, clases_reales):
    """
    Evalúa el rendimiento del clustering comparando las asignaciones que se sacaron
    en el algoritmo con las clases reales, usando la mejor permutación posible de etiquetas.

    Calcula métricas por clase (Precision, Recall, F1) y métricas globales (macro).

    Return: 
        1. aciertos: porcentaje de puntos correctamente clasificados.
        2. macro_precision, macro_recall, macro_f1: métricas promediadas entre clases.
        3. asignaciones_traducidas: asignaciones con etiquetas reales mapeadas.
    """
    clases_unicas = sorted(list(set(clases_reales)))
    clusters_encontrados = sorted(list(set(asignaciones)))

    mejor_mapeo = {}
    max_coincidencias = -1

    # Si hay menos clusters que clases reales, agrega los faltantes para la permutación
    if len(clusters_encontrados) < len(clases_unicas):
        faltantes = set(range(len(clases_unicas))) - set(clusters_encontrados)
        clusters_encontrados.extend(list(faltantes))

    # Prueba todas las permutaciones posibles de etiquetas para encontrar el mejor mapeo
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

    # Traduce las asignaciones del cluster al nombre de clase real según el mejor mapeo
    asignaciones_traducidas = [mejor_mapeo.get(a, -1) for a in asignaciones]

    # Calcula métricas por clase: Precision, Recall y F1-Score
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

        # Calcula métricas evitando divisiones por cero
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

    # Calcula métricas macro (promedio entre todas las clases)
    n_clases = len(clases_unicas)
    macro_precision = (
        sum(m["Precision"] for m in metricas_por_clase.values()) / n_clases
    )
    macro_recall = sum(m["Recall"] for m in metricas_por_clase.values()) / n_clases
    macro_f1 = sum(m["F1-Score"] for m in metricas_por_clase.values()) / n_clases
    aciertos = max_coincidencias / len(clases_reales)

    return aciertos, macro_precision, macro_recall, macro_f1, asignaciones_traducidas


def convertir_a_tabla(puntos_plot, asignaciones_reales, asignaciones_predichas):
    """
    Convierte los datos del análisis en una lista de diccionarios
    lista para ser enviada como JSON.

    Cada elemento contiene las features del punto, su clase real y la predecida.
    """
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


# ─── Configuración de la aplicación Flask ────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir solicitudes desde la página


@app.route("/")
def index():
    """Sirve la página principal de la aplicación."""
    return render_template("index.html")


@app.route("/api/upload-kmeans", methods=["POST"])
def subir_archivo():
    """
    Endpoint principal que recibe un archivo .csv, .data o .txt con datos y ejecuta K-Means.

    Espera un archivo con formato: Feature1, Feature2, ..., Clase
    Parámetros del formulario:
        - archivo: el archivo .csv, .data o .txt con los datos.
        - max_veces: número máximo de iteraciones del algoritmo (por defecto 999)
          o 'auto' que pone también 999.

    Return: 
        1. JSON con los resultados del análisis, incluyendo métricas,
           centroides y la tabla de datos con predicciones.
    """
    try:
        # Verifica que se haya enviado un archivo en la solicitud
        if "archivo" not in request.files:
            return jsonify({"error": "No se proporciono un archivo"}), 400

        archivo = request.files["archivo"]
        max_veces_raw = request.form.get("max_veces", "auto")

        # Determina el número máximo de iteraciones
        if max_veces_raw.strip() == "" or max_veces_raw.strip().lower() == "auto":
            max_veces = 999  # Prácticamente sin límite :p
        else:
            try:
                max_veces = int(max_veces_raw)
            except ValueError:
                max_veces = 999  # Por si hay error en el input pone lo mismo que auto

        # Lee y decodifica el contenido del archivo CSV
        contenido = archivo.read().decode("utf-8").strip().split("\n")

        puntos_plot = []
        clases = []

        # Parsea cada línea del CSV y separa features de la clase
        for linea in contenido:
            if not linea.strip():
                continue  # Ignora líneas vacías

            partes = linea.split(",")
            if len(partes) < 3:
                return jsonify(
                    {
                        "error": "Se requieren al menos 3 columnas (ej. Feature1, Feature2, Clase)"
                    }
                ), 400

            # La última columna es la clase; el resto son features numéricas
            puntos_plot.append([float(x) for x in partes[:-1]])
            clases.append(partes[-1].strip())

        # El número de clusters k es igual al número de clases únicas
        k = len(set(clases))

        # Genera todos los pares de features para analizar combinaciones de 2 dimensiones
        puntos_plot_pares, indices_pares = dividir_lista_en_pares(puntos_plot)
        resultados_analisis = []

        # Ejecuta K-Means para cada par de características
        for index_par, puntos_plot_par in enumerate(puntos_plot_pares):
            cols_comparadas = indices_pares[index_par]

            # Inicializa centroides con K-Means++
            clusters = inicializar_centroides(puntos_plot_par, k)
            lista_asignaciones = []

            # Itera hasta que no cambien los centroides o hasta alcanzar el límite de iteraciones
            for veces in itertools.count(1):
                lista_asignaciones = sacar_distancias(puntos_plot_par, clusters)
                lista_dividida = dividir_en_valor(lista_asignaciones)
                centros_dict = calcular_centros(puntos_plot_par, lista_dividida, k)

                clusters_new = [centros_dict[i] for i in sorted(centros_dict.keys())]

                if veces >= max_veces:
                    break  # Se alcanzó el límite de iteraciones
                elif clusters_new != clusters:
                    clusters = clusters_new  # Los centroides cambiaron, continuar
                else:
                    break  # Los centroides no cambiaron, fin del algoritmo

            # Evalúa el rendimiento del clustering para este par de features
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

        # Ordena los resultados por F1-Score de mayor a menor
        resultados_analisis.sort(key=lambda x: x["f1"], reverse=True)

        # Toma el mejor resultado (el par de features con mayor F1)
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
    # Ejecutar el servidor en modo debug en el puerto 5000
    app.run(debug=True, port=5000)
