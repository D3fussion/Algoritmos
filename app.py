import math
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def distancia_euclidiana(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))


def media_lista(lista_de_listas):
    if not lista_de_listas:
        return []
    dimension = len(lista_de_listas[0])
    suma = [0] * dimension
    for p in lista_de_listas:
        for i in range(dimension):
            suma[i] += p[i]
    return [s / len(lista_de_listas) for s in suma]


def k_means_manual(puntos, k, max_iter=100):
    centroides = random.sample(puntos, k)
    asignaciones = [-1] * len(puntos)

    for _ in range(max_iter):
        cambio = False
        nuevos_clusters = [[] for _ in range(k)]

        for i, punto in enumerate(puntos):
            dists = [distancia_euclidiana(punto, c) for c in centroides]
            cluster_id = dists.index(min(dists))

            if asignaciones[i] != cluster_id:
                asignaciones[i] = cluster_id
                cambio = True

            nuevos_clusters[cluster_id].append(punto)

        for idx in range(k):
            if nuevos_clusters[idx]:
                centroides[idx] = media_lista(nuevos_clusters[idx])

        if not cambio:
            break

    return asignaciones, centroides


def evaluar_precision(asignaciones, etiquetas_reales, k):
    mapa_cluster_etiqueta = {}

    for cluster_id in range(k):
        indices = [i for i, x in enumerate(asignaciones) if x == cluster_id]
        if not indices:
            continue

        conteo_etiquetas = {}
        for idx in indices:
            lbl = etiquetas_reales[idx]
            conteo_etiquetas[lbl] = conteo_etiquetas.get(lbl, 0) + 1

        etiqueta_dominante = max(conteo_etiquetas, key=conteo_etiquetas.get)
        mapa_cluster_etiqueta[cluster_id] = etiqueta_dominante

    aciertos = 0
    predicciones = []

    for i, cluster_id in enumerate(asignaciones):
        pred = mapa_cluster_etiqueta.get(cluster_id, "Desconocido")
        predicciones.append(pred)
        if pred == etiquetas_reales[i]:
            aciertos += 1

    precision = aciertos / len(etiquetas_reales)
    return precision, predicciones, mapa_cluster_etiqueta


@app.route('/api/upload-kmeans', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        n_init = int(request.form.get('n_init', 10))

        contenido = file.read().decode('utf-8').strip().split('\n')

        datos_raw = []
        etiquetas = []
        features_matrix = []

        for linea in contenido:
            partes = linea.strip().split(',')
            if len(partes) < 2: continue

            feats = [float(x) for x in partes[:-1]]
            lbl = partes[-1].strip()

            features_matrix.append(feats)
            etiquetas.append(lbl)

            datos_raw.append({
                'features': feats,
                'class': lbl
            })

        num_features = len(features_matrix[0])
        clases_unicas = list(set(etiquetas))
        k = len(clases_unicas)

        resultados_analisis = []

        for i in range(num_features):
            for j in range(i + 1, num_features):

                puntos_2d = [[fila[i], fila[j]] for fila in features_matrix]
                mejor_precision = -1
                mejor_resultado = None

                for _ in range(n_init):
                    asignaciones, centroides = k_means_manual(puntos_2d, k)
                    precision, preds, mapa = evaluar_precision(asignaciones, etiquetas, k)

                    if precision > mejor_precision:
                        mejor_precision = precision
                        print(f"Asignaciones: {asignaciones}")
                        mejor_resultado = {
                            'indices_features': [i, j],
                            'clases_unicas': len(clases_unicas),
                            'nombres_features': [f'Feature {i + 1}', f'Feature {j + 1}'],
                            'precision': precision,
                            'centroides': centroides,
                            'asignaciones': asignaciones,
                            'predicciones': preds
                        }

                resultados_analisis.append(mejor_resultado)

        resultados_analisis.sort(key=lambda x: x['precision'], reverse=True)

        mejor_modelo = resultados_analisis[0]
        for idx, row in enumerate(datos_raw):
            row['prediccion_kmeans'] = mejor_modelo['predicciones'][idx]

        print(datos_raw)

        return jsonify({
            'total_samples': len(datos_raw),
            'ranking': resultados_analisis,
            'table_data': datos_raw
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)