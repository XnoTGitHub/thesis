import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

# Überprüfe, ob der Dateiname als Parameter übergeben wurde
if len(sys.argv) != 2:
    print("Verwendung: python plot_data.py <Dateiname>")
    sys.exit(1)

# Lese den Dateinamen aus den Kommandozeilenargumenten
file_name = sys.argv[1]

# Lese den gesamten Inhalt der Datei ein
with open(file_name, 'r') as file:
    data_text = file.read()

# Verwende reguläre Ausdrücke, um die relevanten Daten zu extrahieren
pattern = r'Epoch \[(\d+) / \d+\] average reconstruction error: ([\d.]+)   validation error one: ([\d.]+)   validation error two: ([\d.]+)'
matches = re.findall(pattern, data_text)

# Erstelle eine Liste der extrahierten Daten
data_list = [(int(match[0]), float(match[1]), float(match[2]), float(match[3])) for match in matches]

# Erstelle ein DataFrame aus den extrahierten Daten
df = pd.DataFrame(data_list, columns=['Epoch', 'Reconstruction Error', 'Validation Error One', 'Validation Error Two'])

# Erstelle das Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Reconstruction Error'], label='Reconstruction Error', marker='o')
plt.plot(df['Epoch'], df['Validation Error One'], label='Validation Error One', marker='o')
plt.plot(df['Epoch'], df['Validation Error Two'], label='Validation Error Two', marker='o')

# Beschriftungen und Legende hinzufügen
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs. Epoch')
plt.legend()

# Zeige das Plot
plt.grid(True)
plt.show()
