# Cargar librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
img = cv2.imread("./segmentationImages/cat-original.png")
original = img.copy()

# Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Suavizar para evitar ruido y mejorar la separación entre el gato y el fondo
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Detectar bordes (esto ayuda a watershed)
edges = cv2.Canny(blur, 50, 150)
# Convertir bordes a imagen binaria (primer plano aproximado)
ret, thresh = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Invertir si es necesario (gato más claro que fondo)
thresh = cv2.bitwise_not(thresh)
# Limpiar ruido por morfología
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# Fondo seguro (Agrandar áreas negras para asegurar que el fondo esté correctamente definido)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Primer plano seguro 
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform,
                       0.4 * dist_transform.max(),
                       255,
                       0)
sure_fg = np.uint8(sure_fg)
# Zona desconocida
unknown = cv2.subtract(sure_bg, sure_fg)

# Marcadores
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
# Watershed
markers = cv2.watershed(img, markers)
# Marcar bordes en rojo
img[markers == -1] = [0, 0, 255]
# Mostrar resultados
plt.figure(figsize=(15,7))
plt.subplot(231), plt.title("Original"), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.axis("off")
plt.subplot(232), plt.title("Grises"), plt.imshow(gray, cmap='gray'), plt.axis("off")
plt.subplot(233), plt.title("Bordes Canny"), plt.imshow(edges, cmap='gray'), plt.axis("off")
plt.subplot(234), plt.title("Umbral inicial"), plt.imshow(thresh, cmap='gray'), plt.axis("off")
plt.subplot(235), plt.title("Mapa de regiones"), plt.imshow(markers, cmap='nipy_spectral'), plt.axis("off")
plt.subplot(236), plt.title("Watershed final"), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis("off")
plt.show()
