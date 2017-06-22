import re
import lxml
import urllib.request
import urllib
from bs4 import BeautifulSoup
#####资源库，以下两个有下载限制
#ABCwang------------http://www.abcjiaoyu.com/ziyuan/s1/t1/
#获取所有链接http://www.ziyuanku.com/down.aspx?id=149727&eduLevel=1
def download(url):
        print('downloading:',url)
        try:
                html=urllib.request.urlopen(url).read()
        except urllib.request.URLError as e:
                print('download error:',e.reason)
                html=None
        return html

url='http://www.xuekeedu.com/c2-2-1-3-0-1-0-0-0.html?pn=10'
response=urllib.request.urlopen(url)
soup = BeautifulSoup(response, "lxml")

#将链接放入txt中
with open('link_grade1s_math_test.txt', 'a', encoding='utf-8') as f:
 for infor in soup.find_all(class_='docx'):
    link_a=infor.find_all('a' )
    for link in link_a:
        link=link
        print('http://www.xuekeedu.com'+link.get('href'))
        f.write('http://www.xuekeedu.com'+link.get('href')+'\n')
 for infor1 in soup.find_all(class_='doc'):
    link_a = infor1.find_all('a')
    for link in link_a:
        link = link
        print('http://www.xuekeedu.com' + link.get('href'))
        f.write('http://www.xuekeedu.com' + link.get('href') + '\n')
 for infor2 in soup.find_all(class_='pdf'):
    link_a = infor2.find_all('a')
    for link in link_a:
        link = link
        print('http://www.xuekeedu.com' + link.get('href'))
        f.write('http://www.xuekeedu.com' + link.get('href') + '\n')
#想实现自动下载
# for i in range(149744,149800):
#         link='http://www.ziyuanku.com/down.aspx?id='+str(i)+'&eduLevel=1'
#         data=urllib.request.urlopen(link).read()
#         filepath = '../learningresources/grade1s' + str(i) + '.doc'
#         print('download: ', link, ' -> ', filepath, '\n')
#         urllib.request.urlretrieve(link, filepath)





