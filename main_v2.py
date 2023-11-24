import os
import requests
import argparse
import math
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob as tb
from urllib.parse import urlparse

ROOT = os.path.dirname(__file__)

def get_posts(url):

    api_url = url + 'wp-json/wp/v2/posts'
    posts = []
    page = 1

    while True:
        response = requests.get(api_url, params={
            'page': page,
            'per_page': 100
        })

        if not response.status_code == 200:
            break
            
        posts += response.json()
        page += 1
    
    extracted = {}
    for p in posts:
        extracted[datetime.strptime(p['date'], '%Y-%m-%dT%H:%M:%S')] = {
            'link': p['link'],
            'content': p['content']['rendered'],
        }
    
    keys = list(extracted.keys())
    keys.sort(reverse=True)
    extracted = {i: extracted[i] for i in keys}
    
    return extracted

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract relevant keywords using TF-IDF analysis.')
    parser.add_argument('url', help='Supply Wordpress website url.')
    args = parser.parse_args()

    url = args.url
    base = urlparse(url).netloc
    posts = get_posts(url)
    for p in posts:
        content = ''
        soup = BeautifulSoup(posts[p]['content'], 'html.parser')
        for x in soup.find_all('p'):
            content += ' ' + x.text.lower()
        
        posts[p]['content'] = content
    
    list_content = [tb(posts[p]['content']) for p in posts]
    list_words_scores = []
    for p in posts:
        blob = tb(posts[p]['content'])
        scores = {word: tfidf(word, blob, list_content) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:5]:
            list_words_scores.append([posts[p]['link'], word, score])

    df = pd.DataFrame(list_words_scores)
    df.to_excel(os.path.join(ROOT, f'{base} - TFIDF.xlsx'), header=['URL', 'Word', 'TF-IDF score'], index=False)


