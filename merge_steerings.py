import os

def count_lines(file_name):
    with open(file_name, 'r') as f:
        return sum(1 for line in f)

base_dir = './../data1-7/'  # Basisverzeichnis, in dem sich 'params' befindet
params_dir = os.path.join(base_dir, 'params1-7')

# Für jede der Kategorien 'train', 'test' und 'val'...
for category in ['train', 'test', 'val']:
    category_path = os.path.join(params_dir, category)
    output_file_name = base_dir + f"sets/{category}.csv"
    
    with open(output_file_name, 'w') as output_file:
        first_file = True  # um den Header nur aus der ersten Datei zu extrahieren

        # Durchlaufen Sie jedes Unterverzeichnis im Kategorieverzeichnis...
        for town_dir in os.listdir(category_path):
            town_dir_path = os.path.join(category_path, town_dir)
            steering_file_path = os.path.join(town_dir_path, 'steering.txt')

            if os.path.exists(steering_file_path):
                with open(steering_file_path, 'r') as steering_file:
                    # Wenn es die erste Datei ist, kopieren Sie den gesamten Inhalt
                    if first_file:
                        output_file.write(steering_file.read())
                        first_file = False
                    else:
                        # überspringen Sie den Header und kopieren Sie den Rest
                        next(steering_file)  # überspringt den Header
                        output_file.write(steering_file.read())

    # Ausgabe der Zeilenzahl für die aktuelle Datei
    print(f"Anzahl der Zeilen in {output_file_name}: {count_lines(output_file_name)}")

print("Fertig!")

