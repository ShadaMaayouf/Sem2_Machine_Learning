import numpy as np
from sklearn.model_selection import train_test_split

import LR_1_Definition as lr
import LR_0_Dataset as ds
import matplotlib.pyplot as plt

# Ihre Daten
# Datensatz Erstellen
Dring = ds.Dring
z = np.linspace(140, 190, 100)
#Der Parameter test_size=0.2 bedeutet, dass 20% der Daten für den Testdatensatz verwendet werden, während der Rest für den Trainingsdatensatz verwendet wird.
#dieses Datnsatz enthält 10 zeilen, d.h. 2 zufällige Zeilen dieser Zeilen werden für den Testdatensatz verwendet.
Dtrain, Dtest = train_test_split(Dring, test_size=0.2, random_state=42)
#oder man kann Zeilen spezifizieren
#Dtest = Dring.iloc[[7, 8]]  # 2. und 9. Zeile, da die Indexierung bei 0 beginnt
#Dtrain = Dring.drop([7, 8])  # Alle Zeilen außer der 2. und 9.

# Berechnen Sie den Mittelwert der y-Werte und die Gesamtsumme der Quadrate (SST, Nenner in der Formel R^2(D,f)) für den Trainingsdatensatz für den Trainingsdatensatz
y_mean_train = np.mean(Dtrain['y'])
SST_train = np.sum((Dtrain['y'] - y_mean_train)**2)

# Hier sind 3 mögliche lineare Modelle (unterschiedlicher Güte), die die Daten in Dring erklären:
models = [(47, 0.01), (-4, 1 / 3), (9, 0.25), (5.3623451652386365, 0.2686413708690333)]

#Das optimale trainierte Modell ist: h(x) = 5.3623451652386365 + 0.2686413708690333*x

for a, b in models:
    # Berechnen Sie die Vorhersagen für dieses Modell auf dem Trainingsdatensatz
    y_pred_train = lr.f(a, b, Dtrain['x'])

    # Berechnen Sie die Residuenquadratsumme (SSE) für dieses Modell auf dem Trainingsdatensatz
    SSE_train = np.sum((Dtrain['y'] - y_pred_train) ** 2)

    # Berechnen Sie das Bestimmtheitsmaß R^2 für dieses Modell auf dem Trainingsdatensatz
    R2_train = 1 - SSE_train / SST_train

    print(f"Das Bestimmtheitsmaß R^2 für das Modell f(x) = {a} + {b}*x auf dem Trainingsdatensatz beträgt: {R2_train}")

    # Berechnen Sie die Vorhersagen für dieses Modell auf dem Testdatensatz
    y_pred_test = lr.f(a, b, Dtest['x'])

    # Berechnen Sie den Mittelwert der y-Werte für den Testdatensatz
    y_mean_test = np.mean(Dtest['y'])

    # Berechnen Sie die Gesamtsumme der Quadrate (SST) für den Testdatensatz
    SST_test = np.sum((Dtest['y'] - y_mean_test) ** 2)

    # Berechnen Sie die Residuenquadratsumme (SSE) für dieses Modell auf dem Testdatensatz
    SSE_test = np.sum((Dtest['y'] - y_pred_test) ** 2)

    # Berechnen Sie das Bestimmtheitsmaß R^2 für dieses Modell auf dem Testdatensatz
    R2_test = 1 - SSE_test / SST_test

    print(f"Das Bestimmtheitsmaß R^2 für das Modell f(x) = {a} + {b}*x auf dem Testdatensatz beträgt: {R2_test}")
    # Diagramm erstellen
    plt.figure(figsize=(10, 6))
    plt.scatter(Dtrain['x'], Dtrain['y'], label='Trainingsdaten')
    plt.scatter(Dtest['x'], Dtest['y'], color='red', label='Testdaten')
    plt.plot(z, lr.f(a, b, z), label=f'f(x) = {a} + {b}*x')
    plt.title('Visualisierung des Trainings- und Testdatensatzes und des Modells')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()