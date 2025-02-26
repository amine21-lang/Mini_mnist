import torch
import pandas as pd

def generate_digit(digit):
    
    
    matrix = torch.ones((12, 6))  
    
    
    if digit == 0:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
    
    elif digit == 1:
        matrix[1:11, 3] = 0  
        matrix[2, 2] = 0
        matrix[1, 3] = 0
        
    elif digit == 2:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
    elif digit == 3:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
    elif digit == 4:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0]
        ])
        
    elif digit == 5:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
    elif digit == 6:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
    elif digit == 7:
        matrix[1:11, 1:5] *= torch.tensor([
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ])
        
    elif digit == 8:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
        
    elif digit == 9:
        matrix[1:11, 1:5] *= torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ])
    
    return matrix

def add_noise(matrix, noise_level=0.3):
    """
    Ajoute du bruit à la matrice (maintenant 1 = blanc, 0 = noir)
    """
    # Bruit gaussien
    gaussian_noise = torch.randn_like(matrix) * noise_level
    
    # Bruit sel et poivre inversé (1 = blanc, 0 = noir)
    salt_pepper = torch.rand_like(matrix)
    salt = (salt_pepper > 0.97).float()  # ~3% de pixels blancs
    pepper = (salt_pepper < 0.03).float()  # ~3% de pixels noirs
    
    noisy_matrix = matrix + gaussian_noise
    noisy_matrix = torch.where(salt == 1, torch.ones_like(matrix), noisy_matrix)
    noisy_matrix = torch.where(pepper == 1, torch.zeros_like(matrix), noisy_matrix)
    
    # Décalage aléatoire
    if torch.rand(1) > 0.5:
        shift_x = int(torch.randint(-1, 2, (1,)))
        shift_y = int(torch.randint(-1, 2, (1,)))
        noisy_matrix = torch.roll(noisy_matrix, shifts=(shift_x, shift_y), dims=(0, 1))
    
    return torch.clamp(noisy_matrix, 0, 1)

def create_dataset(num_samples_per_digit=100, noise_level=0.3):
    """
    Crée le dataset
    """
    all_samples = []
    all_labels = []
    
    for digit in range(10):
        base_matrix = generate_digit(digit)
        for _ in range(num_samples_per_digit):
            current_noise = noise_level * (0.8 + 0.4 * torch.rand(1).item())
            noisy_matrix = add_noise(base_matrix, current_noise)
            flat_matrix = noisy_matrix.flatten()
            all_samples.append(flat_matrix)
            all_labels.append(digit)
    
    X = torch.stack(all_samples)
    y = torch.tensor(all_labels)
    
    pixel_cols = [f'pixel_{i}' for i in range(72)]
    df = pd.DataFrame(X.numpy(), columns=pixel_cols)
    df['label'] = y.numpy()
    
    return df

if __name__ == "__main__":
    # Générer le dataset
    df = create_dataset(num_samples_per_digit=100, noise_level=0.3)
    df.to_csv('mini_mnist.csv', index=False)
    print(f"Dataset créé avec succès ! Dimensions: {df.shape}")
    print(f"Nombre d'exemples par classe:")
    print(df['label'].value_counts().sort_index())