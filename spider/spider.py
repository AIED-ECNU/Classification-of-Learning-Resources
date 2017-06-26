import re
import lxml
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
# import cookielib
import http.cookiejar
import requests
from lxml import html
import os
#####资源库，以下两个有下载限制
#ABCwang------------http://www.abcjiaoyu.com/ziyuan/s1/t1/
#获取所有链接http://www.ziyuanku.com/down.aspx?id=149727&eduLevel=1
for i in range(1,30):
 url='http://www.xuekeedu.com/c1-2-3-2-0-1-0-0-0.html?pn='+str(i)
 response=urllib.request.urlopen(url)
 soup = BeautifulSoup(response,'lxml')

#将链接放入txt中
 with open('link_grade2s_yuwen_jiaoan.txt', 'a', encoding='utf-8') as f:
   for infor in soup.find_all(class_='docx'):
     link_a=infor.find_all('a' )
     for link in link_a:
        link=link
        title=link.get('title')
        f.write(title+'.docx'+ '\n')
        ID=(link.get('href').replace('/s','')).replace('.html','')
        print(title+'.docx')
        link_each='http://www.xuekeedu.com/Download.aspx?UrlID=10&SoftID=' + ID
        print(link_each)
        f.write(link_each+ '\n')
        # urllib.request.urlretrieve(link_each, '../learningresources/xuekeedu/grade1senglish/doc/'+title.replace('|','') + ".docx")
   for infor1 in soup.find_all(class_='doc'):
     link_a = infor1.find_all('a')
     for link in link_a:
        link = link
        title = link.get('title')
        f.write(title+'.doc'+ '\n')
        ID=(link.get('href').replace('/s','')).replace('.html','')
        print(title+'.doc')
        link_each='http://www.xuekeedu.com/Download.aspx?UrlID=10&SoftID=' + ID
        print(link_each)
        f.write(link_each+ '\n')
        # urllib.request.urlretrieve(link_each,'../learningresources/xuekeedu/grade1senglish/doc/' + title.replace('|','') + ".doc")
   for infor2 in soup.find_all(class_='pdf'):
     link_a = infor2.find_all('a')
     for link in link_a:
        link = link
        title = link.get('title')
        f.write(title+'.pdf'+ '\n')
        ID=(link.get('href').replace('/s','')).replace('.html','')
        print(title+'.pdf')
        link_each='http://www.xuekeedu.com/Download.aspx?UrlID=10&SoftID=' + ID
        print(link_each)
        f.write(link_each+ '\n')
        # urllib.request.urlretrieve(link_each, '../learningresources/xuekeedu/grade1senglish/pdf/' + title.replace('|','') + ".pdf")
   for infor3 in soup.find_all(class_='ppt'):
     link_a = infor3.find_all('a')
     for link in link_a:
        link = link
        title = link.get('title')
        f.write(title+'.ppt'+ '\n')
        ID=(link.get('href').replace('/s','')).replace('.html','')
        print(title+'.ppt')
        link_each='http://www.xuekeedu.com/Download.aspx?UrlID=10&SoftID=' + ID
        print(link_each)
        f.write(link_each+ '\n')
        # urllib.request.urlretrieve(link_each, '../learningresources/xuekeedu/grade1senglish/ppt/' + title.replace('|','') + ".ppt")







