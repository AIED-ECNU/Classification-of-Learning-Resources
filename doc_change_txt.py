# -*- coding: cp936 -*-
#####��doc��docx�ļ�ת��Ϊtxt�ļ���
import win32com
from win32com.client import Dispatch
import os
import fnmatch

all_FileNum = 0
debug = 0

def Translate(path):
    '''
    ��һ��Ŀ¼������doc��docx�ļ�ת��txt
    ��Ŀ¼�´���һ����Ŀ¼TxtFile
    #��Ŀ¼��fileNames.txt����һ���ı��������е�dox\docx�ļ���
    '''
    global debug, all_FileNum
    if debug:
        print(path)

    # ��Ŀ¼�������ļ�������
    files = os.listdir(path)
    # ��Ŀ¼�´���һ����Ŀ¼TxtFile��������ת�����txt�ļ�
    TxtFile = os.path.abspath(os.path.join(path, 'TxtFile'))
    if not os.path.exists(TxtFile):
        os.mkdir(TxtFile)
    if debug:
        print(TxtFile)
    # # ����һ���ı��������е�doc\docx�ļ���
    # fileNames = os.path.abspath(os.path.join(TxtFile, 'fileNames.txt'))
    # o = open(fileNames, "w")
    try:
        for filename in files:
            if debug:
                print(filename)
            # �������doc\docx�ļ�
            if not fnmatch.fnmatch(filename, '*.doc') and not fnmatch.fnmatch(filename, '*.docx'):
                continue
            # �����doc\docx��ʱ�ļ�
            if fnmatch.fnmatch(filename, '~$*'):
                continue
            if debug:
                print(filename)
            # �ļ�·��
            docpath = os.path.abspath(os.path.join(path, filename))

            # �õ�һ���µ��ļ���,��ԭ�ļ����ĺ�׺�ĳ�txt
            # new_txt_name = ''
            if fnmatch.fnmatch(filename, '*.doc'):
                new_txt_name = filename[:-4] + '.txt'
            else:
                new_txt_name = filename[:-5] + '.txt'
            if debug:
                print(new_txt_name)
             #�����ɵ�txt�ļ�����TxtFile �ļ�����
            word_to_txt = os.path.join(os.path.join(TxtFile, new_txt_name))
            print(word_to_txt)

            # word �д���һdoc\docx�ļ�
            wordoffice = win32com.client.Dispatch('Word.Application')
            wordoffice.Visible = 1
            doc = wordoffice.Documents.Open(docpath)

            # Ϊ����python�����ں���������r��ʽ��ȡtxt�Ͳ��������룬����Ϊ4
            doc.SaveAs(word_to_txt,4)
            doc.Close()
            # o.write(word_to_txt + '\n')
            all_FileNum += 1  #�ļ���Ŀ��1
    finally:
        wordoffice.Quit()

if __name__ == '__main__':
    print(
    '''
        ��һ��Ŀ¼������doc��docx�ļ�ת��txt
        ��Ŀ�´���һ����Ŀ¼TxtFile
        ��Ŀ¼��fileNames.txt����һ���ı��������е�dox\docx�ļ���
    ''')
    mypath = '..\\learningresources\\xuekeedu\\grade2syuwen\\doc\\test'
    print('���ɵ��ļ���:')
    Translate(mypath)
    print( 'The Total Files Numbers = ', all_FileNum )





