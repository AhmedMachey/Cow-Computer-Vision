import cv2
import numpy as np

# Charger image
img = cv2.imread(r"C:\\Users\\AHMED\\OneDrive\\Desktop\\projet\\src\\vache5.jpg.png")
h, w = img.shape[:2]

# ------------------------------------------------------------
# 1) Convertir en LAB (très efficace pour animaux)
# ------------------------------------------------------------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Lisser sans détruire les contours
L_filtered = cv2.bilateralFilter(L, 15, 40, 40)

# ------------------------------------------------------------
# 2) Détection automatique par OTSU
# ------------------------------------------------------------
_, mask_dark = cv2.threshold(L_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ------------------------------------------------------------
# 3) Morphologie : fermer trous + garder forme complète
# ------------------------------------------------------------
kernel = np.ones((25,25), np.uint8)
mask_closed = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)
mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

# ------------------------------------------------------------
# 4) Garder le plus grand composant (la vache)
# ------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed)

largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask_big = np.where(labels == largest, 255, 0).astype("uint8")

# ------------------------------------------------------------
# 5) Fermer la forme avec un CONVEX HULL
# ------------------------------------------------------------
contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(contours[0])
mask_hull = np.zeros_like(mask_big)
cv2.drawContours(mask_hull, [hull], -1, 255, -1)

# ------------------------------------------------------------
# 6) GrabCut final (optionnel mais rend parfait)
# ------------------------------------------------------------
mask_gc = np.where(mask_hull == 255, cv2.GC_FGD, cv2.GC_BGD).astype('uint8')
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
cv2.grabCut(img, mask_gc, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

mask_final = np.where((mask_gc==2)|(mask_gc==0), 0, 255).astype('uint8')

# ------------------------------------------------------------
# 7) Dessiner contour + calcul surface
# ------------------------------------------------------------
contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = img.copy()
cv2.drawContours(out, contours_final, -1, (0, 0, 255), 3)

# Surface
surface = np.sum(mask_final == 255)

cv2.putText(out, f"Surface: {surface} px", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# ------------------------------------------------------------
# Affichage
# ------------------------------------------------------------
cv2.imshow("Masque final", mask_final)
cv2.imshow("Vache detectee", out)
cv2.waitKey(0)
