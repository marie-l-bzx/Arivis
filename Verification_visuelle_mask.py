import numpy as np
import napari
from aicsimageio import AICSImage
from pathlib import Path

# -------------------
# Fichiers
# -------------------
infile = Path(r"C:\Users\marie\Documents\Arivis\data\processed\20250225_SCI-9334_18Fev_ZTL_940_D7_Stitch_Linear unmixing.czi")
maskfile = infile.with_suffix(".mask_corrige_TF500.npy")  # même nom, extension .mask.npy

# -------------------
# Charger image CZI
# -------------------
img = AICSImage(infile)
stack = img.get_image_data("CZYX", S=0, T=0)  # shape (C, Z, Y, X)

# Normaliser pour affichage
stack = stack.astype(float)
stack /= stack.max()

# -------------------
# Charger le masque numpy
# -------------------
mask = np.load(maskfile)

# -------------------
# Visualisation Napari
# -------------------
viewer = napari.Viewer()

# Ajouter les canaux
viewer.add_image(stack[0], name="Canal 1 (CFP)", colormap="cyan", blending="additive")
if stack.shape[0] > 1:
    viewer.add_image(stack[1], name="Canal 2 (YFP)", colormap="yellow", blending="additive")
if stack.shape[0] > 2:
    viewer.add_image(stack[2], name="Canal 3 (RFP)", colormap="red", blending="additive")

# Ajouter le masque comme labels
viewer.add_labels(mask.astype(np.uint8), name="Masque Exclusion")

napari.run()
