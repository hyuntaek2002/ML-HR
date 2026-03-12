import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

topics = ["정치", "경제", "사회", "생활/문화", "IT/과학", "세계", "연예", "스포츠"]

def get_full_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if "naver.com" in url:
            article = soup.find(id="dic_area")
            if article: return article.get_text(strip=True)
            
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
    except:
        return ""

def collect_news():
    print("전 분야 뉴스 수집을 시작합니다. (분야별 1건, 총 8건)")
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    
    for topic in topics:
        url = f"https://openapi.naver.com/v1/search/news.json?query={topic}&display=1&sort=sim"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            for item in items:
                full_text = get_full_text(item['link'])
                if not full_text or len(full_text) < 50:
                    full_text = item['description']
                
                data = {
                    "title": item['title'],
                    "description": full_text, 
                    "originallink": item['originallink'],
                    "topic": topic  
                }
                supabase.table("news_data").insert(data).execute()
            print(f"[{topic}] 분야 1건 수집 완료")
        else:
            print(f"[{topic}] 수집 실패")

if __name__ == "__main__":
    collect_news()