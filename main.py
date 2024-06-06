import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Escolha do número de clusters (K)
n_clusters = 3

# 2. Base de dados fixa (Inicialização)
X = np.array([
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0],
    [9, 2], [9, 4], [9, 0]
])

# 3. Aplicar o K-Means (Inicialização dos centróides e Repetição)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)  # Ajusta o modelo aos dados
y_kmeans = kmeans.predict(X)  # Atribuição de pontos aos clusters

# 4. Centróides após convergência
centers = kmeans.cluster_centers_

# 5. Visualizar os resultados
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('Clusters and Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
