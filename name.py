import os
from glob import glob
from random import shuffle

txt_file = glob("/Users/vivekgupta/work/darknet/persondetection_data/labels/*.txt")
total_files = len(txt_file)

txt_file.sort()
shuffle(txt_file)
train_txt = txt_file[0:int(total_files * 0.9)]
test_txt = txt_file[int(total_files * 0.9):]

with open('train.txt','w') as f:
	for each_txt in train_txt:
		path = each_txt.split('.')[0] + '.jpg' + '\n'
		f.write(path)
	f.close()

with open('test.txt','w') as f:
	for each_txt in test_txt:
		path = each_txt.split('.')[0] + '.jpg' + '\n'
		f.write(path)
	f.close()
#count=0
#path = '/Users/vivekgupta/work/darknet/persondetection_data/data1'
#data_folder1='/Users/vivekgupta/work/darknet/persondetection_data/labels1'
#data_folder2='/Users/vivekgupta/work/darknet/persondetection_data/labels2'
#for root, directories, files in os.walk(path, topdown=False):
#	for name in files:
#		name1=name.replace("txt","jpg")
#		count=count+1
#		#print(type(name))
#		if(count<700):
#			print((os.path.join(data_folder1, name1)))
#		else:
#			print((os.path.join(data_folder2, name1)))


