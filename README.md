# ✨ Algoritmos de Segmentación de Imágenes

Este repositorio contiene implementaciones de tres algoritmos fundamentales para la segmentación de imágenes en Python, usando OpenCV y Matplotlib.

---

## 🛠️ Requisitos e Instalación

### Dependencias

Los tres scripts utilizan las mismas librerías principales:

* **NumPy:** Para manejo eficiente de matrices y datos de imagen.
* **OpenCV (`opencv-python`):** Para la lectura y manipulación de imágenes.
* **Matplotlib:** Para la visualización de los resultados.

### Instalación

Instala todas las dependencias con el siguiente comando:

```bash
pip install numpy opencv-python matplotlib
```

---

## 📂 Estructura de Datos

Todas las imágenes de prueba deben colocarse dentro de la carpeta: `segmentationImages/`.

Asegúrate de que la ruta `image_path` dentro de cada script apunte correctamente a tus imágenes.

---

## ▶️ Ejecución de los Algoritmos

A continuación, se detalla cómo ejecutar y configurar cada uno de los tres algoritmos de segmentación.

### 1. Segmentación por Crecimiento de Regiones (Seed Segmentation)

Este algoritmo (`seedSegmentation.py`) detecta todas las regiones de la imagen a partir de semillas, utilizando un criterio de similitud adaptativo basado en la media de la región.

**Fichero:** `seedSegmentation.py`

#### Configuración

Ajusta las siguientes variables en la sección de "Ejemplo de Uso" del archivo:

* **`image_path`**: Ruta a la imagen de entrada (ejemplo: `segmentationImages/gato.png`).
* **`intensity_threshold`**: Valor clave para el algoritmo. Define la tolerancia de similitud para que los píxeles se unan a una región.

#### Comando

```bash
python seedSegmentation.py
```

---

### 2. Segmentación Watershed (Cuencas Hidrográficas)

El algoritmo Watershed (`watershedSegmentation.py`) es eficaz para separar objetos que se tocan o están superpuestos. Trata la imagen como un mapa topográfico donde las intensidades son alturas.

**Fichero:** `watershedSegmentation.py`

#### Configuración

Este algoritmo requiere preprocesamiento (detección de bordes y marcadores internos) que debe configurarse en el código:

* **`image_path`**: Ruta a la imagen de entrada.
* **Parámetros de Preprocesamiento**: Probablemente necesitará ajustar umbralización, filtrado (`kernel_size`) y el manejo de marcadores iniciales.

#### Comando

```bash
python watershedSegmentation.py
```

---

### 3. Segmentación por Corte de Gráficas (Graph Cut)

El método Graph Cut (`graphCutSegmentation.py`) modela la segmentación como un problema de flujo de corte mínimo en un grafo, lo que permite una segmentación óptima para separar un objeto del fondo, generalmente requiriendo que el usuario defina interactivamente las áreas de "primer plano" y "fondo" (a través de marcadores o rectángulos).

**Fichero:** `graphCutSegmentation.py`

#### Configuración

Este algoritmo suele requerir la definición de una región inicial:

* **`image_path`**: Ruta a la imagen de entrada.
* **Área de Interés (ROI)**: Se debe definir el rectángulo inicial que delimita el objeto de interés.

#### Comando

```bash
python graphCutSegmentation.py
```