# Tarea IRIS KMeans

Aplicación web que carga un dataset, ejecuta el algoritmo **K-Means++** sobre todas las combinaciones posibles de 2 features, y muestra un ranking de resultados con gráficas de dispersión y métricas de rendimiento.

---

## Requisitos previos

- [uv](https://docs.astral.sh/uv/) (gestor de entornos de Python)
- Python >= 3.9

Si no tienes `uv` instalado:

**macOS / Linux:**
```bash
brew install uv
```

**Windows:**
```powershell
winget install --id=astral-sh.uv -e
```

---

## Instalación y ejecución

**1. Instala las dependencias:**
```bash
uv sync
```

**2. Inicia el servidor:**
```bash
uv run app.py
```

**3. Abre la aplicación en tu navegador:**
```
http://localhost:5000
```

---

## Cómo usar la aplicación

### 1. Cargar un archivo de datos

- El archivo debe ser `.csv`, `.txt` o `.data`
- Cada línea debe tener el formato: `Feature1, Feature2, ..., Clase`
- La **última columna** siempre es la clase/etiqueta
- Se requieren **al menos 3 columnas** (mínimo 2 features + 1 clase)

Ejemplo de archivo válido:
```
5.1,3.5,1.4,Iris-setosa
4.9,3.0,1.4,Iris-setosa
6.3,3.3,4.7,Iris-versicolor
```

### 2. Configurar las iteraciones

| Opción | Descripción |
| :----------- | :------------ |
| Número | Cantidad máxima de iteraciones del algoritmo (1–999) |
| Auto ✓ | Sin límite práctico (equivale a 999 iteraciones) |

### 3. Analizar

Haz clic en **"Analizar Combinaciones"**. El servidor ejecutará K-Means++ para cada par posible de features y devolverá los resultados ordenados por **F1-Score**.

### 4. Ver resultados

- **Tarjetas de métricas:** F1-Score, Aciertos, Precisión y Recall para cada par de features, junto con su gráfica de dispersión
- **Resumen:** el mejor par de features y el total de muestras procesadas

### 5. Descargar el CSV

Haz clic en el botón **"Exportar CSV"** en la tabla de predicciones. El archivo descargado contiene:

| Columna | Descripción |
| :----------- | :------------ |
| Numero | Índice de la muestra |
| Features | Todos los valores del punto |
| Clase Real | Etiqueta original del dataset |
| Prediccion | Etiqueta asignada por K-Means |
| Estado | `Correcto` o `Incorrecto` |

---

## Dependencias

| Paquete | Versión mínima |
| :----------- | :------------ |
| flask | 3.1.3 |
| flask-cors | 6.0.2 |
