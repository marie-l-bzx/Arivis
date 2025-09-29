import torch

# Test 1: Vérification basique
print("PyTorch version:", torch.__version__)  # Doit contenir "+cu121"
print("CUDA disponible:", torch.cuda.is_available())  # Doit retourner True
print("Nom du GPU:", torch.cuda.get_device_name(0))  # Doit afficher "NVIDIA GeForce RTX 3050"

# Test 2: Opération sur GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    y = torch.matmul(x, x)
    print("Opération sur GPU:", y.device)  # Doit afficher "cuda:0"
else:
    print("ERREUR: CUDA n'est pas disponible.")
