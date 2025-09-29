import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from aicsimageio import AICSImage
from skimage.transform import resize
import napari
#from torchvision.transforms import functional as F
#from einops import rearrange
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Désactive l'erreur (mais pas le conflit)

# =============================
# Vérification initiale du GPU
# =============================
### MODIF: Ajout des diagnostics GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("ATTENTION: PyTorch n'utilise pas le GPU !")
    device = torch.device("cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de: {device}")
# =============================
# Chemin image et masque
# =============================
czi_train = Path("C:/Users/marie/Documents/Arivis/20250303_SCI-9322_24Fev_ZTL_940_D7_Stitch_Linear unmixing.czi")
mask_train = Path("C:/Users/marie/Documents/Arivis/20250303_SCI-9322_24Fev_ZTL_940_D7_Stitch_Linear unmixing.npy")

# =============================
# Chargement des données
# =============================
img = AICSImage(czi_train)
stack = img.get_image_data("CZYX", S=0, T=0)[:3]  # shape (C, Z, Y, X)
stack = stack.astype(np.float32) / stack.max()
stack = stack.transpose(1, 2, 3, 0)  # shape (Z, Y, X, C)

mask = np.load(mask_train)[..., np.newaxis]  # shape (Z, Y, X, 1)

IMG_SIZE = (256, 256)
X_train = np.array([resize(im, IMG_SIZE, preserve_range=True) for im in stack])
y_train = np.array([resize(m, IMG_SIZE, preserve_range=True) for m in mask])
y_train = (y_train > 0.5).astype(np.float32)

# =============================
# Vision Transformer pour la segmentation (version optimisée)
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True  # Pour une API plus intuitive (B, N, C)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # Self-Attention
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn

        # MLP
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=384):  ### MODIF: embed_dim réduit à 384
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ViTSegmentation(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, n_classes=1,
                 embed_dim=384, depth=6, n_heads=6, mlp_ratio=2.):
        super().__init__()
        # ### MODIF: Stockez tous les paramètres nécessaires
        self.img_size = img_size  # <-- Correction principale
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Patch Embedding (utilise img_size)
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Nombre de patches
        self.n_patches = self.patch_embed.n_patches

        # Token CLS (optionnel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Transformer Encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Head de segmentation
        self.segmentation_head = nn.Sequential(
    nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),  # (B, 192, 256, 256)
    nn.GELU(),
    nn.Conv2d(embed_dim // 2, n_classes, kernel_size=1)   # (B, n_classes, 256, 256)
)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Ajout du CLS token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches+1, embed_dim)

        # Ajout du positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Extraction des patches (on ignore le CLS token)
        x = x[:, 1:, :]  # (B, n_patches, embed_dim)

        # Reshape pour la segmentation
        h = w = int(self.n_patches ** 0.5)  # Racine carrée du nombre de patches
        x = x.permute(0, 2, 1)  # (B, embed_dim, n_patches)
        x = x.reshape(x.shape[0], self.embed_dim, h, w)

        # Interpolation pour retrouver la résolution originale
        x = nn.functional.interpolate(
            x,
            size=(self.img_size, self.img_size),  # ### MODIF: Utilise self.img_size
            mode='bilinear',
            align_corners=True
        )

        # Head de segmentation
        x = self.segmentation_head(x)
        return x

# =============================
# Initialisation du modèle et optimiseur
# =============================
model = ViTSegmentation().to(device)  ### MODIF: .to(device) pour s'assurer que le modèle est sur GPU
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # Équivalent à BinaryCrossEntropy + sigmoid

# =============================
# Préparation des données
# =============================
X_train_tensor = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
y_train_tensor = torch.from_numpy(y_train.transpose(0, 3, 1, 2)).float()
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# =============================
# Boucle d'entraînement avec diagnostics GPU
# =============================
model.train()
for epoch in range(500):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        # ### MODIF: Diagnostics GPU en temps réel
        if epoch % 50 == 0 and i == 0:  # Affiche seulement au premier batch de chaque 50e epoch
            print(f"\n--- Diagnostics GPU (Epoch {epoch}) ---")
            print(f"Inputs device: {inputs.device}")
            print(f"Mémoire GPU allouée: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            print(f"Mémoire GPU maximale: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Vérification que les outputs sont bien sur GPU
        if epoch % 50 == 0 and i == 0:
            print(f"Outputs device: {outputs.device}")

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # ### MODIF: Libération de la mémoire GPU après chaque epoch
    torch.cuda.empty_cache()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader):.4f}')


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
model.eval()
with torch.no_grad():
    for z in range(stack_val.shape[0]):
        im = resize(stack_val[z], IMG_SIZE, preserve_range=True)[np.newaxis, ...]
        im_tensor = torch.from_numpy(im.transpose(0, 3, 1, 2)).float().to(device)
        pred_mask = model(im_tensor)[0, 0].cpu().numpy() > 0.5
        masks_pred[z] = resize(pred_mask.astype(np.uint8), stack_val.shape[1:3], preserve_range=True)

# -----------------
# Correction dans Napari
# -----------------
viewer = napari.Viewer()
for c in range(3):
    viewer.add_image(stack_val[..., c], name=f"Canal {c+1}", colormap=["cyan","green","red"][c],
                    blending="additive", contrast_limits=[0,1])

labels_layer = viewer.add_labels(masks_pred, name="Masque Prédit")

print("Corrige le masque dans Napari, puis ferme la fenêtre pour sauvegarder.")
napari.run()

# Sauvegarder le masque corrigé
corrected_mask_path = val_file.with_suffix(".mask_corrige_ViT500.npy")
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

# Convertir en tenseurs PyTorch
X_train_new_tensor = torch.from_numpy(X_train_new.transpose(0, 3, 1, 2)).float()
y_train_new_tensor = torch.from_numpy(y_train_new.transpose(0, 3, 1, 2)).float()

new_dataset = TensorDataset(X_train_new_tensor, y_train_new_tensor)
new_dataloader = DataLoader(new_dataset, batch_size=4, shuffle=True)

# Réentraîner le modèle
model.train()
for epoch in range(20):
    running_loss = 0.0
    for inputs, targets in new_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Fine-tuning Epoch {epoch}, Loss: {running_loss/len(new_dataloader)}')

# -----------------
# Prédiction sur une nouvelle image de validation avec le modèle mis à jour
# -----------------
val_files = [
    Path(r"C:\Users\marie\Documents\Arivis\data\processed\20250224_SCI-9341_17Fev_ZTL_940_D7_Stitch_Linear unmixing.czi"),
]

model.eval()
with torch.no_grad():
    for val_file in val_files:
        print(f"Prédiction sur {val_file.name}")
        img_val = AICSImage(val_file)
        stack_val = img_val.get_image_data("CZYX", S=0, T=0)[:3]
        stack_val = stack_val.astype(np.float32)/stack_val.max()
        stack_val = stack_val.transpose(1,2,3,0)

        masks_pred = np.zeros(stack_val.shape[:3], dtype=np.uint8)
        for z in range(stack_val.shape[0]):
            im = resize(stack_val[z], IMG_SIZE, preserve_range=True)[np.newaxis, ...]
            im_tensor = torch.from_numpy(im.transpose(0, 3, 1, 2)).float().to(device)
            pred_mask = model(im_tensor)[0, 0].cpu().numpy() > 0.5
            masks_pred[z] = resize(pred_mask.astype(np.uint8), stack_val.shape[1:3], preserve_range=True)

        # Visualisation facultative pour vérification
        viewer = napari.Viewer()
        for c in range(3):
            viewer.add_image(stack_val[...,c], name=f"Canal {c+1}",
                            colormap=["cyan","yellow","red"][c],
                            blending="additive", contrast_limits=[0,1])
        viewer.add_labels(masks_pred, name="Masque Prédit")
        napari.run()
