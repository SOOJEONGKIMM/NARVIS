from tqdm import tqdm
import time

import numpy as np
import json
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# 데이터 불러오기
data_path = './news_dataset.json'
data = json.load(open(data_path))

df = pd.DataFrame(data)
contents = df['content']

# stopwords
stopwords_path = './stopwords-ko.txt'    # source: https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a#file-stopwords-ko-txt
with open(stopwords_path,'r', encoding='utf-8') as file:
    stopwords = [line.split()[0] for line in file]

# tfidf calculation
vector = TfidfVectorizer(stop_words=stopwords, max_features=1000)
tfidf = vector.fit_transform(contents)
print('TF-IDF 행렬 크기 :',tfidf.shape)

path='./remove_similar/'
pickle.dump(vector, open(path+'vector', 'wb')) 
pickle.dump(tfidf, open(path+'/tfidf', 'wb')) 


# similarity가 threshold를 넘는 비슷한 기사들의 index를 가져옴. (최대 50개)
def find_similar(tfidf, index, threshold):
    cosine_similarities = linear_kernel(tfidf[index:index+1], tfidf).flatten()    
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    indices_to_be_removed = [i for i in related_docs_indices if cosine_similarities[i] > threshold]
    return indices_to_be_removed[:50]

# dataset에서 같은 날짜에 올라온 비슷한 기사들 삭제
def remove_similar(similar_articles, index):
    for j in similar_articles:
        if j > index:
            try: 
                a_dates = data[index]['dates'][:10]
                if a_dates == data[j]['dates'][:10]:
                    data[j]=''
            except:
                continue



def main():

    print('starts removing similar articles')
    start_time = time.time()

    for i in tqdm(range(len(data))):
        similar_articles = find_similar(tfidf, i, threshold=0.5)     # 제거할 기사들의 index
        remove_similar(similar_articles, i)
    
    print("---%s seconds ---" % (time.time() - start_time))
        
    output_path = './news_remove_similar_articles.json'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent='\t')


if __name__ == '__main__':
    main()