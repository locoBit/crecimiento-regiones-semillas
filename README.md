# 🚀 Crecimiento de Regiones Basado en Semillas (Global)

Este proyecto implementa la **Versión 3** del algoritmo de **Crecimiento de Regiones Multisemilla (Global)**. Su objetivo es segmentar automáticamente todas las regiones de una imagen en escala de grises, utilizando un criterio adaptativo (basado en la media de la región) y visualizando los puntos de inicio (semillas) de cada región detectada.

---

## 🛠️ Requisitos del Sistema

* **Python 3.x**
* **Gestor de paquetes `pip`**

---

## 📦 Instalación de Dependencias

El proyecto requiere las siguientes librerías principales de Python:

1. **NumPy:** Para el manejo eficiente de matrices de imagen.
2. **OpenCV (`opencv-python`):** Para la lectura de imágenes.
3. **Matplotlib:** Para la visualización de los resultados.

Instala todas las dependencias con el siguiente comando:

```bash
pip install numpy opencv-python matplotlib
```

---

## ▶️ Ejecución del Proyecto

1. **Guarda tu Imagen:**
    Coloca la imagen que deseas segmentar (e.g., `gato.png`) en el mismo directorio que el script (`semillas.py`) o utiliza su ruta completa.

2. **Configura las Variables:**
    Abre el script y ajusta las variables de configuración en la sección de "Ejemplo de Uso":

    * **`image_path`**: La ruta a tu archivo de imagen.
    * **`intensity_threshold`**: El valor que define la **similitud de intensidad** para el crecimiento de la región. **Ajustar este valor es crucial** para obtener buenos resultados (un rango típico de prueba es entre 5 y 30).

    ```python
    image_path = 'gato.png' 
    intensity_threshold = 15 # Ajusta este valor según tu imagen
    ```

3. **Corre el Script:**
    Ejecuta el proyecto desde tu terminal:

    ```bash
    python semillas.py
    ```

El script imprimirá el número total de regiones detectadas y mostrará (o guardará, si usas el modo `Agg`) dos gráficas: la imagen original con las semillas marcadas en rojo, y el mapa de regiones segmentadas con un código de colores.

---

## 🛑 Solución de Errores Frecuentes

### 1. `ModuleNotFoundError: No module named 'cv2'` o `'matplotlib'`

Asegúrate de haber ejecutado correctamente la instalación de dependencias (`pip install ...`). Si estás usando un **entorno virtual**, verifica que el entorno esté activado.

### 2. `ModuleNotFoundError: No module named '_tkinter'`

Este error ocurre cuando **Matplotlib** no puede encontrar las librerías necesarias para dibujar las ventanas gráficas interactivas (`TkAgg`).

**Solución Recomendada (Modo Sin Ventana):**

Si la solución de reinstalación de Python no funciona, evita el modo interactivo configurando Matplotlib para que guarde el resultado en un archivo:

1. **Añade esta línea al inicio del script, antes de `import matplotlib.pyplot as plt`:**
    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```
2. **Reemplaza `plt.show()`** en la parte de visualización con la siguiente línea para guardar el resultado:
    ```python
    plt.savefig('output_segmentacion_resultado.png')
    plt.close()
    ```# crecimiento-regiones-semillas
