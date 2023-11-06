import pandas as pd
import csv
import os

sets = ['train','val','test']

for version in sets:

  # Define the directory path
  directory_path = "././../data1-7.2/params1-7/" + version + "/"

  # Initialize an empty list to store subdirectories
  array_of_dirs = []

  # Iterate through all items (files and directories) in the specified path
  for item in os.listdir(directory_path):
      item_path = os.path.join(directory_path, item)#, "pred_depth/")
      
      # Check if the item is a directory
      if os.path.isdir(item_path):
        array_of_dirs.append(item_path)
  print(array_of_dirs)
  csv_file_lines = []  


  for data_dir in array_of_dirs:
    #for camera_view_dir in os.listdir(data_dir):
      if(os.path.isfile(data_dir  + '/steering.txt')):
        df = pd.read_csv(os.path.join(data_dir, 'steering.txt'))
        csv_entries = df.values.tolist()
        print(csv_entries)
        #if camera_view_dir[0] == '1' or camera_view_dir[0] == '7': #if the car is at a traffic light it mostly can't be seen
        #  csv_entries = [[data_dir + camera_view_dir + '/' + name, thr, ste ] for name, thr, ste in csv_entries if thr > 0.06]
        #else:
        #  csv_entries = [[data_dir + camera_view_dir + '/' + name, thr, ste ] for name, thr, ste in csv_entries]
        csv_entries = [[name, ste] for name, throttle, ste in csv_entries]
        csv_file_lines.extend(csv_entries)
      else:
        print('ERROR file ' + data_dir  + '/steering.txt' + ' not found!')

  print(len(csv_entries))

  print(len(csv_file_lines))
  print(csv_file_lines[0])

  csv_header = ['frame_name', 'steer']

  file_path = './../data1-7.2/sets/' + version + '.csv'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'w', newline='') as f:

    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_file_lines)
