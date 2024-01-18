import numpy as np
import LR_1_Definition as lr
import LR_0_Dataset as ds
import matplotlib.pyplot as plt

# Ihre Daten
# Datensatz Erstellen
Dring = ds.Dring

# Datensatz visualisieren
lr.visualise_datensatz(Dring)

#Hier sind 3 mögliche lineare Modelle (unterschiedlicher Güte), die die Daten in Dring erklären:
# Erzeugen Sie Werte für z
z = np.linspace(140, 190, 100)
#Funktionen der entsprechenden linearen Modelle aufgespannt auf den raum z
f1 = lr.f(47,0.01,z)
f2 = lr.f(-4,1/3,z)
f3 = lr.f(9,0.25,z)

#y-werte der linearen Modelle basierend auf Datensatz
y = lr.f(47,0.01,Dring['x'])
y2 = lr.f(-4,1/3,Dring['x'])
y3 = lr.f(9,0.25,Dring['x'])

# Berechnen Sie den quadratischen Fehler für f1 und f2 und f3
L_f1 = lr.L_berechnen(y, Dring['y'])
L_f2 = lr.L_berechnen(y2, Dring['y'])
L_f3 = lr.L_berechnen(y3, Dring['y'])

#Diagramm erstellen:
plt.figure(figsize=(10, 6))
plt.scatter(Dring['x'], Dring['y'], label='Datenpunkte')
plt.plot(z, f1, color='blue', label='f1(z) = 47 + 0.01*z, Quadratischer Fehler = ' +str(L_f1))
plt.plot(z, f2, color='red', label='f2(z) = -4 + 1/3*z, Quadratischer Fehler = '+str(L_f2))
plt.plot(z, f3, color='green', label='f3(z) = 9 + 1/4*z, Quadratischer Fehler = '+str(L_f3))
plt.title('Visualisierung des Datensatzes Dring und der Funktionen f1, f2 und f3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#Nun berechnen wir das OPTIMALE lineare Modell
a,b = lr.lr_berechnen(Dring[['x']], Dring['y'])
foptimal = lr.f(a,b,z)

yoptimal = lr.f(a,b,Dring['x'])
#und dessen quadratischen Fehler L
L_foptimal = lr.L_berechnen(yoptimal, Dring['y'])
#dann visualisieren wir diesen
plt.figure(figsize=(10, 6))
plt.scatter(Dring['x'], Dring['y'], label='Datenpunkte')
plt.plot(z, foptimal, color='blue', label='foptimal(z) = ' +str(a) + ' + ' + str(b) +'z, Quadratischer Fehler = ' +str(L_foptimal))
plt.title('Visualisierung des Datensatzes Dring und der Funktion foptimal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
