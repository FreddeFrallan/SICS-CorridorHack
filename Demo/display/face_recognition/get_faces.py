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




def get_hashes(company, w, quiet=True):
    to_worker, from_worker, poll_worker = w
    
    import os
    import sys
    import cv2
    import tempfile
    import time
            
    images = get_images(company)

    hashes = {}

    temp_name = next(tempfile._get_candidate_names())
    fn = f'/tmp/facerecog.{temp_name}'
    
    for k in images:
        with open(fn, 'wb') as f: f.write(images[k])
        img = cv2.imread(fn)
        os.remove(fn)

        to_worker(('img', img))
        while poll_worker() == False:
            if not quiet: print('.',end='', file=sys.stderr)
            time.sleep(0.01)
        if not quiet: print(file=sys.stderr)
        result = from_worker()
        if len(result) == 1:
            face_hash = result[0][2]
            yield k, face_hash
        else:
            if not quiet: print(f'Failed: {k}', file=sys.stderr)


    return hashes


if __name__ == '__main__':
    import face_recog
    w = face_recog.start_worker(face_recog.worker)
    to_worker, from_worker, poll_worker = w

    for k, h in get_hashes('sics', w):
        print(k)
        print(h.tolist())
        to_worker(('newface', (h, k)))
        
