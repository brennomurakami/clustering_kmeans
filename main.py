import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Gerar dados aleatórios
np.random.seed(42)  # Para garantir que os resultados sejam reproduzíveis
n_samples = 400

# 1. Escolha do número de clusters (K)
n_clusters = 3

# 2. Base de dados fixa (Inicialização)
X = np.random.rand(n_samples, 2) * 10  # Coordenadas aleatórias entre 0 e 10

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

# Calcular e imprimir o número recomendado de clusters (método do cotovelo)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

diff_inertias = np.diff(inertias, n=2)
recommended_clusters = np.argmax(diff_inertias) + 2  # Adicionamos 2 devido à diferenciação
print("Número recomendado de clusters:", recommended_clusters)

plt.show()
