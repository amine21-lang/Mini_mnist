import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

def load_and_prepare_data(file_path='mini_mnist.csv'):
    """
    Charge et prépare les f.onnées
    """
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Séparer features et labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    return X, y, df

def visualize_data_distribution(df):
    """
    Visualise la distribution des classes et statistiques de base
    """
    plt.figure(figsize=(15, 5))
    
    # Distribution des classes
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='label')
    plt.title('Distribution des classes')
    plt.xlabel('Chiffre')
    plt.ylabel('Nombre d\'exemples')
    
    # Statistiques des pixels
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df.drop('label', axis=1))
    plt.title('Distribution des valeurs de pixels')
    plt.xlabel('Pixels')
    plt.xticks([])
    
    plt.tight_layout()
    plt.show()

def analyze_variance_and_noise(X, y):
    """
    Analyse la variance et le bruit dans les données
    """
    # Calculer la variance moyenne par classe
    variances = []
    for digit in range(10):
        digit_samples = X[y == digit]
        var = np.var(digit_samples, axis=0).mean()
        variances.append(var)
    
    plt.figure(figsize=(15, 5))
    
    # Variance par classe
    plt.subplot(1, 2, 1)
    plt.bar(range(10), variances)
    plt.title('Variance moyenne par classe')
    plt.xlabel('Chiffre')
    plt.ylabel('Variance moyenne')
    
    # Distribution des valeurs de pixels
    plt.subplot(1, 2, 2)
    plt.hist(X.flatten(), bins=50)
    plt.title('Distribution des valeurs de pixels')
    plt.xlabel('Valeur du pixel')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.show()
    
    return np.mean(variances)

def perform_clustering(X, y, n_clusters=10):
    """
    Effectue un clustering K-means et analyse les résultats
    """
    # Réduire la dimensionnalité pour la visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Visualiser les résultats
    plt.figure(figsize=(15, 5))
    
    # Clustering vs vraies classes
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10')
    plt.title('Résultats du clustering')
    plt.colorbar(scatter)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10')
    plt.title('Vraies classes')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Calculer la pureté des clusters
    cluster_purity = calculate_cluster_purity(y, cluster_labels)
    return cluster_purity

def calculate_cluster_purity(true_labels, cluster_labels):
    """
    Calcule la pureté des clusters
    """
    contingency_matrix = pd.crosstab(cluster_labels, true_labels)
    return np.sum(np.max(contingency_matrix, axis=1)) / len(true_labels)

def split_and_analyze_data(X, y, test_size=0.2):
    """
    Divise les données en ensembles d'entraînement et de test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Analyser la répartition des classes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=10, alpha=0.5, label='Train')
    plt.hist(y_test, bins=10, alpha=0.5, label='Test')
    plt.title('Distribution des classes dans les ensembles Train/Test')
    plt.xlabel('Chiffre')
    plt.ylabel('Nombre d\'exemples')
    plt.legend()
    
    # Statistiques de base
    plt.subplot(1, 2, 2)
    plt.boxplot([X_train.flatten(), X_test.flatten()], labels=['Train', 'Test'])
    plt.title('Distribution des valeurs de pixels Train vs Test')
    
    plt.tight_layout()
    plt.show()
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Analyse complète du dataset
    """
    print("1. Chargement et préparation des données...")
    X, y, df = load_and_prepare_data()
    
    print("\n2. Visualisation de la distribution des données...")
    visualize_data_distribution(df)
    
    print("\n3. Analyse de la variance et du bruit...")
    mean_variance = analyze_variance_and_noise(X, y)
    print(f"Variance moyenne globale: {mean_variance:.4f}")
    
    print("\n4. Clustering des données...")
    cluster_purity = perform_clustering(X, y)
    print(f"Pureté des clusters: {cluster_purity:.4f}")
    
    print("\n5. Division et analyse des ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = split_and_analyze_data(X, y)
    print(f"Taille de l'ensemble d'entraînement: {len(X_train)}")
    print(f"Taille de l'ensemble de test: {len(X_test)}")

if __name__ == "__main__":
    main()