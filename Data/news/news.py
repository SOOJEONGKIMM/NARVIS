import requests
from bs4 import BeautifulSoup

import re
import datetime
from tqdm import tqdm
from datetime import datetime

import json
import csv

import multiprocessing
import time


# 변수 세팅
corp_file = './dart_corpCodes.csv'   # 기업 리스트 파일
output_path = './output/'            # output 파일 경로

max_page = 100    # 네이버 뉴스 몇 페이지를 가져올 건지
years = 3         # 몇 년치 뉴스를 가져올 건지
listed = True     # 상장기업 or 비상장기업



#회사 목록 리스트에서 상장사, 비상장사 구분하여 가져옴
def get_company_list(file_path, listed):

    with open(file_path,'r', encoding='utf-8') as f:
        mycsv=csv.reader(f)
        companies=[]
        for row in mycsv:
            company={}
            company['corp_code']=row[0]
            company['corp_name']=row[1]
            company['stock_code']=row[2]
            company['modify_date']=row[3]
            companies.append(company)
        
    cor_listed=[]
    cor_not_listed=[]
    for company in companies:
        if company['stock_code'] == ' ':
            cor_not_listed.append(company)
        else:
            cor_listed.append(company)
    
    if listed:
        return cor_listed
    else:
        return cor_not_listed        
       

# 각 기업에 대해서 크롤링 할 최대 페이지 및 기간에 해당하는 뉴스 링크를 가져옴
def news_crawler(query, max_page):
    
    start_pg = 1
    end_pg = (int(max_page)-1)*10+1 

    naver_urls = []   
    
    while start_pg < end_pg:
        
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + query + "&start=" + str(start_pg)
        headers = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36" }
        
        raw = requests.get(url, headers=headers)
        cont = raw.content
        html = BeautifulSoup(cont, 'html.parser')
        
        for urls in html.select("a.info"):
            try:
                if "news.naver.com" in urls['href']:
                    naver_urls.append(urls['href'])              
            except Exception as e:
                continue
        
        start_pg += 10
        
    return naver_urls 


# 각 기업에 대해 뉴스 data 생성
def newsdataset(query):
    
    total_news = []

    total_urls = news_crawler(query, max_page)
    total_urls = list(set(total_urls)) 

    headers = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36" }

    for url in (total_urls):
        try:
            raw = requests.get(url,headers=headers)
            html = BeautifulSoup(raw.text, "html.parser")
        except requests.exceptions.RequestException:
            continue
        
        news={}
        pattern1 = '<[^>]*>'

        ## 날짜
        try:
            html_date = html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))
       
        start_year = datetime.now().year - years
        try:
            news_year = int(news_date[:4])
            if news_year < start_year:
                continue
            else:
                news['dates']=news_date
        except ValueError:
            continue

        # url
        news['url']=url

        # 뉴스 제목
        title = html.select("div#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        title = ''.join(str(title))
        title = re.sub(pattern=pattern1,repl='',string=title)
        news['titles']=title

        #뉴스 본문
        content = html.select("div#dic_area")
        content = ''.join(str(content))
        content = re.sub(pattern=pattern1,repl='',string=content)
        pattern2 = '\n'
        content = re.sub(pattern=pattern2,repl='',string=content)
        pattern3 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern3,'')
        news['content']=content

        total_news.append(news)
            
    # return total_news
    path = output_path + query + '.json'       
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(total_news, f, ensure_ascii=False, indent='\t')


def main():

    start_time = time.time()

    companies = get_company_list(corp_file, listed)
    queries=[companies[i]['corp_name'] for i in range(1,len(companies))]

    with multiprocessing.Pool(4) as pool:
        list(tqdm(pool.imap(newsdataset, queries), total=len(queries)))
    pool.close()
    pool.join()

    print("---%s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    main()