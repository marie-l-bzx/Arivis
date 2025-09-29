import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from aicsimageio import AICSImage 
from skimage.transform import resize
import napari


# -----------------
# Chemin image et masque
# -----------------
czi_train = Path("C:/Users/marie/Documents/Arivis/20250303_SCI-9322_24Fev_ZTL_940_D7_Stitch_Linear unmixing.czi")
mask_train = Path("C:/Users/marie/Documents/Arivis/20250303_SCI-9322_24Fev_ZTL_940_D7_Stitch_Linear unmixing.npy")

# -----------------
# Chargement des données
# -----------------
img = AICSImage(czi_train)
stack = img.get_image_data("CZYX", S=0, T=0)[:3]  # shape (C, Z, Y, X)
stack = stack.astype(np.float32) / stack.max()
stack = stack.transpose(1, 2, 3, 0)  # shape (Z, Y, X, C)

mask = np.load(mask_train)[..., np.newaxis]  # shape (Z, Y, X, 1)

IMG_SIZE = (256, 256)
X_train = np.array([resize(im, IMG_SIZE, preserve_range=True) for im in stack])
y_train = np.array([resize(m, IMG_SIZE, preserve_range=True) for m in mask])
y_train = (y_train > 0.5).astype(np.float32)

# -----------------
# U-Net simplifié
# -----------------
def unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    c1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D(2)(c1)
    
    c2 = Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D(2)(c2)
    
    c3 = Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
    
    u2 = Conv2DTranspose(64, 2, strides=2, padding="same")(c3)
    u2 = concatenate([u2, c2])
    c4 = Conv2D(64, 3, activation="relu", padding="same")(u2)
    c4 = Conv2D(64, 3, activation="relu", padding="same")(c4)
    
    u1 = Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    u1 = concatenate([u1, c1])
    c5 = Conv2D(32, 3, activation="relu", padding="same")(u1)
    c5 = Conv2D(32, 3, activation="relu", padding="same")(c5)
    
    outputs = Conv2D(1, 1, activation="sigmoid")(c5)
    model = Model(inputs, outputs)
    return model

model = unet(input_shape=X_train.shape[1:])
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------
# Entraînement initial
# -----------------
model.fit(X_train, y_train, batch_size=4, epochs=50, validation_split=0.1)


# -----------------
# Image de validation à corriger
# -----------------

val_file = Path(r"C:\Users\marie\Documents\Arivis\data\processed\20250225_SCI-9334_18Fev_ZTL_940_D7_Stitch_Linear unmixing.czi")
img_val = AICSImage(val_file)
stack_val = img_val.get_image_data("CZYX", S=0, T=0)[:3]  # C, Z, Y, X
stack_val = stack_val.astype(np.float32) / stack_val.max()
stack_val = stack_val.transpose(1, 2, 3, 0)  # Z, Y, X, C

# Créer un masque prédictif 3D
masks_pred = np.zeros(stack_val.shape[:3], dtype=np.uint8)

# Prédiction slice par slice
for z in range(stack_val.shape[0]):
    im = resize(stack_val[z], IMG_SIZE, preserve_range=True)[np.newaxis, ...]
    pred_mask = model.predict(im)[0, ..., 0] > 0.5
    masks_pred[z] = resize(pred_mask.astype(np.uint8), stack_val.shape[1:3], preserve_range=True)

# -----------------
# Correction dans Napari
# -----------------
viewer = napari.Viewer()
for c in range(3):
    viewer.add_image(stack_val[..., c], name=f"Canal {c+1}", colormap=["cyan","green","red"][c], blending="additive", contrast_limits=[0,1])

labels_layer = viewer.add_labels(masks_pred, name="Masque Prédit")

print("Corrige le masque dans Napari, puis ferme la fenêtre pour sauvegarder.")
napari.run()

# Sauvegarder le masque corrigé
corrected_mask_path = val_file.with_suffix(".mask_corrige.npy")
np.save(corrected_mask_path, labels_layer.data)
print(f"Masque corrigé sauvegardé : {corrected_mask_path}")

# -----------------
# Réapprentissage avec l'image corrigée
# -----------------
# Redimensionner le stack et le masque corrigé pour le modèle
X_corrige = np.array([resize(im, IMG_SIZE, preserve_range=True) for im in stack_val])
y_corrige = np.array([resize(m, IMG_SIZE, preserve_range=True) for m in labels_layer.data[..., np.newaxis]])
y_corrige = (y_corrige > 0.5).astype(np.float32)

# Ajouter au dataset initial
X_train_new = np.concatenate([X_train, X_corrige], axis=0)
y_train_new = np.concatenate([y_train, y_corrige], axis=0)

# Réentraîner le modèle
model.fit(X_train_new, y_train_new, batch_size=4, epochs=20)

# -----------------
# Prédiction sur une nouvelle image de validation avec le modèle mis à jour
# -----------------
val_file = [
    Path(r"C:\Users\marie\Documents\Arivis\data\processed\20250224_SCI-9341_17Fev_ZTL_940_D7_Stitch_Linear unmixing.czi"),
]

for val_file in val_file:
    print(f"Prédiction sur {val_file.name}")
    img_val = AICSImage(val_file)
    stack_val = img_val.get_image_data("CZYX", S=0, T=0)[:3]
    stack_val = stack_val.astype(np.float32)/stack_val.max()
    stack_val = stack_val.transpose(1,2,3,0)

    masks_pred = np.zeros(stack_val.shape[:3], dtype=np.uint8)
    for z in range(stack_val.shape[0]):
        im = resize(stack_val[z], IMG_SIZE, preserve_range=True)[np.newaxis,...]
        pred_mask = model.predict(im)[0,...,0] > 0.5
        masks_pred[z] = resize(pred_mask.astype(np.uint8), stack_val.shape[1:3], preserve_range=True)

    # Visualisation facultative pour vérification
    viewer = napari.Viewer()
    for c in range(3):
        viewer.add_image(stack_val[...,c], name=f"Canal {c+1}", colormap=["cyan","yellow","red"][c], blending="additive", contrast_limits=[0,1])
    viewer.add_labels(masks_pred, name="Masque Prédit")
    napari.run()
    
   
