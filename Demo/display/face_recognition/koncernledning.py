import requests
import bs4

url = 'https://www.ri.se/sv/om-rise/organisation/koncernledning'

t = bs4.BeautifulSoup(requests.get(url).content)

im = [x for x in t.findAll('img')]
im = [x for x in im if 'profile_image' in x['src']]

def name(x): return x.parent.parent.find('h2').text

for i in im:
    u = i['src']
    print(name(i), i['src'])
    jpg = requests.get(u).content
    with open(name(i), 'wb') as f:
        f.write(jpg)
    
print('ok')
