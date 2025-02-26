import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Configuration et analyse préliminaire
print("="*50)
print("CONFIGURATION DU RÉSEAU DE NEURONES")
print("="*50)

# Dimensions du problème
INPUT_SIZE = 72  # 12x6 pixels
OUTPUT_SIZE = 10  # 10 chiffres (0-9)
HIDDEN_SIZES = [20, 10]  # Deux couches cachées

print("\nDimensions :")
print(f"- Entrée : {INPUT_SIZE} neurones (12x6 pixels)")
for i, size in enumerate(HIDDEN_SIZES, 1):
    print(f"- Couche cachée {i} : {size} neurones")
print(f"- Sortie : {OUTPUT_SIZE} neurones (10 classes)")

# Calcul théorique du nombre de paramètres
def calculate_theoretical_params():
    total_params = 0
    prev_size = INPUT_SIZE
    
    # Paramètres des couches cachées
    for hidden_size in HIDDEN_SIZES:
        weights = prev_size * hidden_size
        biases = hidden_size
        total_params += weights + biases
        prev_size = hidden_size
    
    # Paramètres de la couche de sortie
    final_weights = prev_size * OUTPUT_SIZE
    final_biases = OUTPUT_SIZE
    total_params += final_weights + final_biases
    
    return total_params

theoretical_params = calculate_theoretical_params()
print(f"\nNombre total de paramètres (théorique) : {theoretical_params:,}")

# Hyperparamètres
print("\nHyperparamètres :")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
RANDOM_SEED = 42

print(f"- Taille du batch : {BATCH_SIZE}")
print(f"- Taux d'apprentissage : {LEARNING_RATE}")
print(f"- Nombre d'époques : {EPOCHS}")
print(f"- Graine aléatoire : {RANDOM_SEED}")

# Configuration de l'optimisation
print("\nConfiguration de l'optimisation :")
print("- Optimiseur : Adam")
print("  └ Paramètres par défaut : β1=0.9, β2=0.999, ε=1e-8")
print("- Fonction de perte : Cross Entropy")
print("- Fonctions d'activation :")
print("  └ Couches cachées : ReLU")
print("  └ Couche de sortie : Softmax")

# Fixer les graines aléatoires
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MiniMNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNet, self).__init__()
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Chargement des données
print("\nChargement et préparation des données...")
df = pd.read_csv('mini_mnist.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Affichage de la répartition des données
print("\nRépartition des classes :")
unique, counts = np.unique(y, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"Chiffre {digit}: {count} exemples")

# Division des données
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"\nTaille des ensembles :")
print(f"- Entraînement : {len(X_train)} exemples")
print(f"- Validation : {len(X_val)} exemples")

# Création des DataLoaders
train_dataset = MiniMNISTDataset(X_train, y_train)
val_dataset = MiniMNISTDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Création et affichage du modèle
model = FullyConnectedNet(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
print("\nArchitecture détaillée du modèle :")
print("="*50)
print(model)
print("\nNombre de paramètres par couche :")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} paramètres")
print(f"Total : {count_parameters(model):,} paramètres")

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Fonctions pour l'entraînement et l'évaluation
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# Entraînement du modèle
print("\nDébut de l'entraînement...")
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

# Visualisation des résultats
plt.figure(figsize=(15, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()

# Courbe de précision
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Visualisation de quelques prédictions
def visualize_predictions(model, val_loader, num_examples=5):
    model.eval()
    images, labels = next(iter(val_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for i in range(num_examples):
        img = images[i].numpy().reshape(12, 6)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Prédit: {predicted[i]}\nVrai: {labels[i]}')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(model, val_loader)