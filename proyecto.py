# biblioteca para abrir el archivo de datos
import pandas as pd
# biblioteca para generar gráficos
import matplotlib.pyplot as plt
# bibliotecas necesarias para balancear el contenido
from sklearn.utils import resample
import numpy as np

# abrir el archivo
dataset = pd.read_csv('ProyectoI_AI\Datasets\diabetes.csv')
print(dataset.to_string())

# crear un gráfico de distribución del ancho de las hojas
dataset["Outcome"].plot(kind='kde').set_xlabel("Cantidad por valores")
plt.ylabel("Densidad")
plt.title("Distribución del balanceo")

# incluir todos los features
X = dataset.iloc[:, 0:8]
# incluir solo los labels
y = dataset.iloc[:, 8]

# seleccionar casos positivos y negativos
positivo = X[y==1]
negativo = X[y==0]

# DownSample a negativo (clase mayor)
negativo_sampled = resample(negativo, replace=False,n_samples=len(positivo), random_state=42)

# juntar los datos X positivos y negativos sampleados
X_sampled = pd.concat([positivo,negativo_sampled])

# juntar datos Y correspondientes a positivos y negativos sampleados
Y_Pos = pd.DataFrame(np.ones((len(positivo), 1)))
Y_Neg = pd.DataFrame(np.zeros((len(negativo_sampled), 1)))
Y_sampled = pd.concat([Y_Pos, Y_Neg])

print(len(X_sampled),len(Y_sampled))

# crear un gráfico de distribución del ancho de las hojas
Y_sampled.plot(kind='kde').set_xlabel("Valores")
plt.ylabel("Densidad")
plt.title("Distribución del balanceo post sampleo")