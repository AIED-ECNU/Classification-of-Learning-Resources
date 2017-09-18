# -*- coding: cp936 -*-
#####将doc，docx文件转换为txt文件中
import win32com
from win32com.client import Dispatch
import os
import fnmatch

all_FileNum = 0
debug = 0

def Translate(path):
    '''
    将一个目录下所有doc和docx文件转成txt
    该目录下创建一个新目录TxtFile
    #新目录下fileNames.txt创建一个文本存入所有的dox\docx文件名
    '''
    global debug, all_FileNum
    if debug:
        print(path)

    # 该目录下所有文件的名字
    files = os.listdir(path)
    # 该目录下创建一个新目录TxtFile，用来放转化后的txt文件
    TxtFile = os.path.abspath(os.path.join(path, 'TxtFile'))
    if not os.path.exists(TxtFile):
        os.mkdir(TxtFile)
    if debug:
        print(TxtFile)
    # # 创建一个文本存入所有的doc\docx文件名
    # fileNames = os.path.abspath(os.path.join(TxtFile, 'fileNames.txt'))
    # o = open(fileNames, "w")
    try:
        for filename in files:
            if debug:
                print(filename)
            # 如果不是doc\docx文件
            if not fnmatch.fnmatch(filename, '*.doc') and not fnmatch.fnmatch(filename, '*.docx'):
                continue
            # 如果是doc\docx临时文件
            if fnmatch.fnmatch(filename, '~$*'):
                continue
            if debug:
                print(filename)
            # 文件路径
            docpath = os.path.abspath(os.path.join(path, filename))

            # 得到一个新的文件名,把原文件名的后缀改成txt
            # new_txt_name = ''
            if fnmatch.fnmatch(filename, '*.doc'):
                new_txt_name = filename[:-4] + '.txt'
            else:
                new_txt_name = filename[:-5] + '.txt'
            if debug:
                print(new_txt_name)
             #将生成的txt文件放在TxtFile 文件夹中
            word_to_txt = os.path.join(os.path.join(TxtFile, new_txt_name))
            print(word_to_txt)

            # word 中打开这一doc\docx文件
            wordoffice = win32com.client.Dispatch('Word.Application')
            wordoffice.Visible = 1
            doc = wordoffice.Documents.Open(docpath)

            # 为了让python可以在后续操作中r方式读取txt和不产生乱码，参数为4
            doc.SaveAs(word_to_txt,4)
            doc.Close()
            # o.write(word_to_txt + '\n')
            all_FileNum += 1  #文件数目加1
    finally:
        wordoffice.Quit()

if __name__ == '__main__':
    print(
    '''
        将一个目录下所有doc和docx文件转成txt
        该目下创建一个新目录TxtFile
        新目录下fileNames.txt创建一个文本存入所有的dox\docx文件名
    ''')
    mypath = '..\\learningresources\\xuekeedu\\grade2syuwen\\doc\\test'
    print('生成的文件有:')
    Translate(mypath)
    print( 'The Total Files Numbers = ', all_FileNum )





