import numpy as np
import pandas as pd

# Gerar dados aleatórios
np.random.seed(42)  # Para garantir que os resultados sejam reproduzíveis
n_samples = 200
X = np.random.rand(n_samples, 2) * 10  # Coordenadas aleatórias entre 0 e 10

# Criar DataFrame do Pandas
df = pd.DataFrame(X, columns=['X', 'Y'])

# Salvar DataFrame como arquivo CSV
df.to_csv('data.csv', index=False)

print("Arquivo 'dados.csv' gerado com sucesso!")