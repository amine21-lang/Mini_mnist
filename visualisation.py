import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_digit(data, index=None, title=None):
    """
    Visualise un seul chiffre du dataset
    """
    if index is not None:
        # Sélectionner une ligne spécifique
        pixel_data = data.iloc[index, :-1]  # Tous sauf le label
        label = data.iloc[index, -1]
    else:
        # Utiliser les données passées directement
        pixel_data = data[:-1]
        label = data[-1]
    
    # Reshape en 12x6
    image = pixel_data.values.reshape(12, 6)
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    else:
        plt.title(f'Chiffre: {int(label)}')

def visualize_sample_digits(csv_file, samples_per_digit=3):
    """
    Visualise plusieurs exemples de chaque chiffre
    """
    # Charger le dataset
    df = pd.read_csv(csv_file)
    
    # Créer une figure avec des sous-plots
    fig, axes = plt.subplots(10, samples_per_digit, figsize=(12, 20))
    fig.suptitle('Échantillons de Mini-MNIST', fontsize=16)
    
    for digit in range(10):
        # Sélectionner les exemples pour ce chiffre
        digit_samples = df[df['label'] == digit].sample(n=samples_per_digit)
        
        for j in range(samples_per_digit):
            plt.sca(axes[digit, j])
            visualize_digit(digit_samples.iloc[j], title=f'Chiffre {digit}' if j == 0 else None)
    
    plt.tight_layout()
    plt.show()

def visualize_average_digits(csv_file):
    """
    Visualise la forme moyenne de chaque chiffre
    """
    # Charger le dataset
    df = pd.read_csv(csv_file)
    
    # Créer une figure avec des sous-plots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Formes moyennes des chiffres', fontsize=16)
    
    for digit in range(10):
        # Calculer la moyenne pour ce chiffre
        digit_samples = df[df['label'] == digit].iloc[:, :-1]
        average_digit = digit_samples.mean().values.reshape(12, 6)
        
        # Afficher dans le sous-plot approprié
        plt.sca(axes[digit // 5, digit % 5])
        plt.imshow(average_digit, cmap='gray')
        plt.axis('off')
        plt.title(f'Moyenne {digit}')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Fonction principale pour visualiser les résultats
    """
    csv_file = 'mini_mnist.csv'
    
    print("1. Visualisation d'échantillons de chaque chiffre...")
    visualize_sample_digits(csv_file)
    
    print("\n2. Visualisation des formes moyennes...")
    visualize_average_digits(csv_file)

if __name__ == "__main__":
    main()