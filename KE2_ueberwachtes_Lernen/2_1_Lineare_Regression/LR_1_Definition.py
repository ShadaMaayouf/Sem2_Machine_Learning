import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

#Funktionen
def visualise_datensatz(df):
    # Erstellen Sie das Diagramm
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['y'])
    plt.title('Visualisierung des Datensatzes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def f(a,b,z):
    return a + b*z

def visualise_f(a,b,z):
    # Berechnen Sie die Werte von f1
    y = f(a,b,z)
    # Erstellen Sie das Diagramm
    plt.figure(figsize=(10, 6))
    plt.plot(z, y, label='f(z) = '+ str(a) +' + '+str(b)+'z')
    plt.title('Visualisierung der Funktion f1')
    plt.xlabel('z')
    plt.ylabel('f(z)')
    plt.legend()
    plt.grid(True)
    plt.show()

def lr_berechnen(X_D, y_D):
    # Erstellen Sie das lineare Regressionsmodell
    model = LinearRegression()
    # Trainieren Sie das Modell mit Ihren Daten
    model.fit(X_D, y_D)
    # Die Koeffizienten des Modells
    # Modell ist eine Funktion der Form h(x) = θ0 + θ1*x, wobei θ0 und θ1 die gelernten Parameter des Modells sind.
    theta_0 = model.intercept_
    theta_1 = model.coef_[0]
    print(f"Das optimale trainierte Modell ist: h(x) = {theta_0} + {theta_1}*x")
    return theta_0, theta_1

def L_berechnen(y, y_D):
    # Berechnet den quadratischen Fehler (MSE: Mean Squared Error)
    L = np.sum((y - y_D) ** 2)
    print(f"Der quadratische Fehler des Modells ist: {L}")
    return L

def L_minimierung_Gradientenabstieg(X_D, y_D):
    # Initialisieren Sie die Parameter auf Null
    theta = np.zeros(2)

    # Setzen Sie die Lernrate und die Anzahl der Iterationen
    alpha = 0.0001
    num_iters = 1000

    # Die Anzahl der Beobachtungen
    m = len(y_D)

    # Führen Sie die Gradientenabstiegsschleife aus
    for i in range(num_iters):
        # Berechnen Sie die Vorhersagen
        y_pred = np.dot(X_D, theta)

        # Berechnen Sie den Fehler
        error = y_pred - y_D

        # Aktualisieren Sie die Parameter
        theta = theta - (alpha / m) * np.dot(X_D.T, error)

    print(f"Die optimalen Parameter sind: {theta}")

