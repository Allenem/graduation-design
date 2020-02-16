import os
import time
import datetime

# The sum of elements in celeba test data recognition & not recognition ≠ the number of celeba test data.
# So I code this to detect repeat elements.
# a:celeba test data; b:recognition; c:not recognition
# There are 5 cases:
# 1.in b+c not in a
# 2.in a not in b+c
# 3.in b and in c
# 4.repeat elements in b
# 5.repeat elements in c

start_time = time.clock()
a = []
b = []
c = []

a_path = 'D:/Celeba/test'
a = os.listdir(a_path)
# for aa in a:
#   with open('C:/Users/puyao/Desktop/a.txt', "a", encoding="utf-8") as fi:
#     fi.write(aa+'\n')

b_path = 'D:/Celeba_face/test'
b_lists = os.listdir(b_path)
for b_list in b_lists:
  (filename, extension) = os.path.splitext(b_list)
  new_filename = filename[3:]
  b.append(new_filename+extension)
  # with open('C:/Users/puyao/Desktop/b.txt', "a", encoding="utf-8") as fi:
  #   fi.write(new_filename+extension+'\n')

c_path = 'C:/Users/puyao/Desktop/nofound.txt'

with open(c_path, encoding="utf-8") as fi:
    lines = fi.readlines()
for line in lines:
  c_name = line[:-1]
  c.append(c_name)

d = b + c

d1 = [x for x in d if x not in a]    # in b+c not in a
d2 = [x for x in a if x not in d]    # in a not in b+c
d3 = [x for x in b if x in c]        # in b and in c


new_b = []
repeat_b = []
new_c = []
repeat_c = []

for i in b:
  if i not in new_b:
    new_b.append(i)
  else:
    repeat_b.append(i)               # repeat elements in b

for i in c:
  if i not in new_c:
    new_c.append(i)
  else:
    repeat_c.append(i)               # repeat elements in c


print(len(a))
print(len(d))
print('d1 =',d1)
print('d2 =',d2)
print('d3 =',d3)
print('repeat_b =',repeat_b)
print('repeat_c =',repeat_c)
end_time = time.clock()
delta_time = datetime.timedelta(seconds = (end_time-start_time))
print('Running time : %s '%(delta_time))


# Finally I found repeat elements in c. Output:
# 20261
# 20269
# d1 = []
# d2 = []
# d3 = []
# repeat_b = []
# repeat_c = ['000120.jpg', '000189.jpg', '000749.jpg', '001344.jpg', '001642.jpg', '001643.jpg', '001652.jpg', '001704.jpg']
# Running time : 0:00:12.015545