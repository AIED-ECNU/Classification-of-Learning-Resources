import re
import lxml
import urllib.request
import urllib
from bs4 import BeautifulSoup
import string

url='http://www.hjszy.com/kejian/yuwen/snj/index.html'
response=urllib.request.urlopen(url)
soup = BeautifulSoup(response, "lxml")

#将链接放入txt中
# with open('link_grade1x_math_test.txt', 'a', encoding='utf-8') as f:
with open('link_grade3_yuwen_kejian.txt', 'a', encoding='utf-8') as f:
 for infor in soup.find_all(class_='box2'):
    links_a=infor.find_all('a' )
    for links in links_a:
        links='http://www.hjszy.com'+links.get('href')#第一层链接
        response1 = urllib.request.urlopen(links)
        soup1 = BeautifulSoup(response1, "lxml")
        link_pos=soup1.find('div',{'class':'zoom'})
        link_each=link_pos.find('a')
        title=link_each.get('title')
        print(title)
        f.write(title+'\n')
        link_each1='http://www.hjszy.com'+link_each.get('href')#第二层链接
        response2 = urllib.request.urlopen(link_each1)
        soup2 = BeautifulSoup(response2, "lxml")
        if "../../" in soup2.find('a').get('href'):
            strc=soup2.find('a').get('href')
            link_each2='http://www.hjszy.com/e/'+strc.replace('../../','')
            print(link_each2)
            f.write(link_each2+ '\n')
            urllib.request.urlretrieve(link_each2,'../learningresources/hjszy/grade2/original/'+title+".rar")
        # http://www.hjszy.com/e/enews/?enews=DownSoft&classid=62&id=10262&pathid=0&pass=c1a2d433e1a743f1366650bc6f5cb318&p=:::
        # f.write('http://www.xuekeedu.com'+link.get('href')+'\n')




