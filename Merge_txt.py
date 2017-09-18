# -*- coding: utf-8 -*-
import os
#文件夹路径
#  语文1-6 年级转完了格式
path= '..\\learningresources\\xuekeedu\\grade3senglish\\doc\\jiaoan\\TxtFile'
filedir=os.path.abspath(path)
#当前文件夹的所有txt文件名称
filenames_all=os.listdir(filedir)
count=0

f=open(path+'\\grade3s_english_jiaoan.txt','w')
# 读取每个txt文件内容到新建的txt中
for filename in filenames_all:
 if( filename!="grade3s_english_jiaoan.txt"):
    filepath=filedir+'/'+filename
    count=count+1
    print(filepath)
    for line in open(filepath ,encoding='gbk',errors='ignore'):
        f.writelines(line)
    f.write('\n')
 else:
     continue
#有问题，都有1G了
f.close()
print(count)


