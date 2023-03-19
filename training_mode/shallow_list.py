import os
import sys

def get_log_path_dict(log_path,fo):
	# log_path = "D:/Graduation/Face"
	count=0
	for root, dirs, files in os.walk(log_path):
		name_list=list()
		# log_path_dict = dict()
		for dir_name in dirs:
			log_path2 = os.path.join(root, dir_name)
			# print(log_path)
			for root2, dirs2, files2 in os.walk(log_path2):
				flag=0
				two=0
				for onefile in files2:
					dir_path = root2[len(log_path)+1:]+'/'+onefile
					# name_list.append([dir_path,count])
					fo.write(dir_path)
					fo.write(' ')
					fo.write(str(count))
					fo.write('\n')
					flag=1
					two+=1 
					if(two>=2):
						break
				count+=flag
			if(count>=100000):
				break
			# log_path_dict[dir_name] = log_path
		return 0# name_list

if __name__ == '__main__':
    print('Done!')
    img_info_file = 'webface_shallow_train_list.txt'
    fo = open(img_info_file, "w")	
    img_list=get_log_path_dict('../data/CASIA-WebFace',fo)
    '''
    for s in img_list:
        fo.write(s[0])
        fo.write(' ')
        fo.write(s[1])
        fo.write('\n')
	'''
    fo.close()
    print('Done!')