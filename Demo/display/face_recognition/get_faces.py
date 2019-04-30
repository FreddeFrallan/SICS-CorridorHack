import requests
import bs4
import multiprocessing as mp

def download(x):
    key, url = x
    return key, requests.get(url).content

def get_images(company):
    assert company in ['sics', 'acreo']

    url = f'https://www.{company}.se/contact/people/all'
    b = bs4.BeautifulSoup(requests.get(url).content, features="html.parser")
    x = dict([(x.parent.text.strip() ,x['src']) for x in b.findAll('img') if 'small_contact' in x['src'] and not 'user-green' in x['src']])

    images = dict(mp.Pool().imap_unordered(download, x.items()))

    return images

for k in get_images('sics'):
    print(k)

        
