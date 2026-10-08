import cv2
import numpy as np

# --------------------------------------------------------
# 1) Charger l'image
# --------------------------------------------------------
image_path = r"C:\Users\AHMED\OneDrive\Desktop\projet\src\vache1.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image introuvable.")

h, w = img.shape[:2]

original = img.copy()

# --------------------------------------------------------
# 2) Prétraitement
# --------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blur, 30, 120)

kernel = np.ones((7, 7), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# --------------------------------------------------------
# 3) Trouver contour principal
# --------------------------------------------------------
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# --------------------------------------------------------
# 4) Masque (silhouette exacte)
# --------------------------------------------------------
mask = np.zeros((h, w), np.uint8)
cv2.drawContours(mask, [largest], -1, 255, -1)

vache_silhouette = cv2.bitwise_and(original, original, mask=mask)

# --------------------------------------------------------
# 5) Surface en pixels
# --------------------------------------------------------
surface_pixels = cv2.contourArea(largest)
print("Surface de la vache (px²) :", surface_pixels)

# --------------------------------------------------------
# 6) Contour polygonal lissé
# --------------------------------------------------------
epsilon = 0.01 * cv2.arcLength(largest, True)
poly = cv2.approxPolyDP(largest, epsilon, True)

img_poly = original.copy()
cv2.drawContours(img_poly, [poly], -1, (0, 0, 255), 3)

# --------------------------------------------------------
# 7) BBOX pour hauteur + longueur
# --------------------------------------------------------
x, y, w_box, h_box = cv2.boundingRect(largest)
hauteur_px = h_box
longueur_px = w_box

print("Hauteur (px) :", hauteur_px)
print("Longueur (px) :", longueur_px)

img_bbox = original.copy()
cv2.rectangle(img_bbox, (x, y), (x+w_box, y+h_box), (0, 255, 0), 3)

# --------------------------------------------------------
# 8) PNG transparent en mémoire (pas enregistré)
# --------------------------------------------------------
rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = mask  # alpha = masque

# --------------------------------------------------------
# 9) Poids estimé
# --------------------------------------------------------
k = 0.012  # coefficient empirique
poids_estime = k * (surface_pixels ** 0.62)

print("Poids estimé (approx) :", round(poids_estime, 1), "kg")

# --------------------------------------------------------
# 10) Affichage
# --------------------------------------------------------
cv2.imshow("Original", original)
cv2.imshow("Silhouette exacte", vache_silhouette)
cv2.imshow("Contour polygonal", img_poly)
cv2.imshow("Bounding Box", img_bbox)
cv2.imshow("Mask", mask)
cv2.imshow("Edges", edges)
cv2.imshow("PNG Transparent (preview)", rgba)

cv2.waitKey(0)
cv2.destroyAllWindows()
