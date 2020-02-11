import os
import time
import datetime

start_time =time.clock()

file_path = 'D:/test'
file_lists = os.listdir(file_path)
for file_list in file_lists:
  (filename, extension) = os.path.splitext(file_list)
  # delete last 2 word
  new_filename = filename[:-2]
  old_file = file_path+'/'+file_list
  new_file = file_path+'/'+new_filename+extension
  os.rename(old_file,new_file) 

end_time = time.clock()
delta_time = datetime.timedelta(seconds = (end_time-start_time))
print('Running time : %s '%(delta_time))

# rename 38970 images output
# Running time : 0:00:39.003402