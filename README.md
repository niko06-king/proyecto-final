import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing  # Dataset alternativo 

# Configuración de gráficos 
sns.set_theme(style="whitegrid", palette="husl")  # Usa "whitegrid", "darkgrid", etc.

# Cargar los datos
california = fetch_california_housing()
datos = pd.DataFrame(california.data, columns=california.feature_names)
datos['VALOR_CASA'] = california.target  

# Explorar los datos
print("\nPrimeras filas:")
print(datos.head())

print("\nResumen estadístico:")
print(datos.describe())

# Gráficos importantes
plt.figure(figsize=(10, 6))
sns.histplot(datos['VALOR_CASA'], bins=30, kde=True)
plt.title('Distribución de precios de viviendas (California)')
plt.xlabel('Precio (en cientos de miles $)')
plt.ylabel('Cantidad')
plt.show()

# Relación de variables
variables = ['MedInc', 'AveRooms', 'Population']  
for var in variables:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=datos[var], y=datos['VALOR_CASA'])
    plt.title(f'Relación entre {var} y el precio')
    plt.show()

# Correlación
correlaciones = datos.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones[['VALOR_CASA']].sort_values('VALOR_CASA', ascending=False), 
            annot=True, cmap='coolwarm')
plt.title('Correlación con el precio de viviendas')
plt.show()
