import urllib.request
import urllib.parse
import re
#打开url
def open_url(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 '
                                 '(KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36')
    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')
    return html
#提取网页内容
def get_content(num):
    # 存放段子的列表
    text_list = []
    for page in range(3, int(num)):
        address = 'http://xiaoyu.pep.com.cn/xxsx_176803/xxsxwd/201705/t20170517_189322' + str(page)+'.shtml'
        html = open_url(address)
        content_a = r'<p>(.*?)</p>'
        result_a = re.findall(content_a, html, re.S | re.M)
        text_list.append(result_a)
    return text_list # 每一页的result都是一个列表，将里面的内容加入到text_list

#对网页内容进一步处理，去掉<b>或<br/>，存入txt内
def save(num):
    with open('a.txt', 'w', encoding='utf-8') as f:
        for each in get_content(num):
            for element in each:
                if '<b>' or '</b>' in element:
                    new_element = re.sub(r'<b>', '\n', element)
                    new_element = re.sub(r'</b>', '\n', new_element)
                    if '<br />' in new_element:
                        # 替换成换行符并输出
                        f.write(new_element)
                        # 没有就照常输出
                    else:
                        f.write(str(new_element) + '\n')
                else:
                    f.write(element)

#调用函数
save(5)

