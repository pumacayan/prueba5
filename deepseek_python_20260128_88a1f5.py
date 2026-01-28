import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos
url = "https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv"
df = pd.read_csv(url)

# Reemplazar ceros por NaN en columnas donde cero no es válido
cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_con_ceros] = df[cols_con_ceros].replace(0, np.nan)

# Imputar valores faltantes con la mediana
for col in cols_con_ceros:
    df[col].fillna(df[col].median(), inplace=True)

# Separar características y variable objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Estandarizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)