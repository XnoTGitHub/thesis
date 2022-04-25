import pandas as pd
import csv
import os
#array_of_dirs = ['data_var_hight_small/Town01_0/pred_depth/','data_var_hight_small/Town01_1/pred_depth/',
#		'data_var_hight_small/Town01_2/pred_depth/','data_var_hight_small/Town01_3/pred_depth/',
#		'data_var_hight_small/Town02_0/pred_depth/','data_var_hight_small/Town02_1/pred_depth/',
#		'data_var_hight_small/Town02_2/pred_depth/','data_var_hight_small/Town02_3/pred_depth/',
#		'data_var_hight_small/Town03_0/pred_depth/','data_var_hight_small/Town04_0/pred_depth/',
#		'data_var_hight_small/Town05_0/pred_depth/','data_var_hight_small/Town06_0/pred_depth/']
array_of_dirs = ['data_var_hight/Town07_0/pred_depth/','data_var_hight/Town07_1/pred_depth/']
#array_of_dirs = ['data_no_traficlights_small/Town01_0/pred_depth/','data_no_traficlights_small/Town01_1/pred_depth/',
#                'data_no_traficlights_small/Town01_2/pred_depth/','data_no_traficlights_small/Town01_3/pred_depth/',
#                'data_no_traficlights_small/Town02_0/pred_depth/','data_no_traficlights_small/Town02_1/pred_depth/',
#                'data_no_traficlights_small/Town02_2/pred_depth/','data_no_traficlights_small/Town02_3/pred_depth/',
#                'data_no_traficlights_small/Town03_0/pred_depth/','data_no_traficlights_small/Town04_0/pred_depth/',
#                'data_no_traficlights_small/Town05_0/pred_depth/','data_no_traficlights_small/Town06_0/pred_depth/']

csv_file_lines = []  


for data_dir in array_of_dirs:
  for camera_view_dir in os.listdir(data_dir):
    if(os.path.isfile(data_dir + camera_view_dir + '/control.csv')):
      df = pd.read_csv(os.path.join(data_dir + camera_view_dir + '/', 'control.csv'))
      csv_entries = df.values.tolist()
      print(csv_entries)
      if camera_view_dir[0] == '1' or camera_view_dir[0] == '7': #if the car is at a traffic light it mostly can't be seen
        csv_entries = [[data_dir + camera_view_dir + '/' + name, thr, ste ] for name, thr, ste in csv_entries if thr > 0.06]
      else:
        csv_entries = [[data_dir + camera_view_dir + '/' + name, thr, ste ] for name, thr, ste in csv_entries]
      csv_file_lines.extend(csv_entries)
    else:
      print('ERROR file ' + data_dir + camera_view_dir + '/control.csv' + ' not found!')

print(len(csv_entries))

print(len(csv_file_lines))
print(csv_file_lines[0])

csv_header = ['frame_name', 'throttle', 'steer']

with open('sets/7_No_Traficlights.csv', 'w', newline='') as f:

  writer = csv.writer(f)
  writer.writerow(csv_header)
  writer.writerows(csv_file_lines)
