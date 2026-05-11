import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

# 환경 설정 및 DB 연결
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 수집할 8개 분야 정의
TOPICS = ["IT", "경제", "사회", "정치", "연예", "스포츠", "생활/문화", "세계"]

def get_full_text(url):
    """뉴스 URL에서 본문 전체 텍스트를 추출"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스 특화 크롤링
        if "naver.com" in url:
            article = soup.find(id="dic_area")
            if article: return article.get_text(strip=True)
            
        # 일반 뉴스 크롤링 (p 태그 기반)
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        return content
    except:
        return ""

def collect_news(topic, count=1):
    """특정 분야(topic)의 뉴스를 수집하여 DB에 저장"""
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    
    url = f"https://openapi.naver.com/v1/search/news.json?query={topic}&display={count}&sort=sim"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        items = response.json().get('items', [])
        for item in items:
            full_text = get_full_text(item['link'])
            if not full_text or len(full_text) < 50:
                full_text = item['description']
            
            # 제목 태그 제거 및 데이터 구성
            clean_title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
            
            data = {
                "title": clean_title,
                "description": full_text, 
                "originallink": item['originallink'],
                "topic": topic  # 8개 분야 중 해당 분야 이름 저장
            }
            
            try:
                supabase.table("news_data").insert(data).execute()
                print(f"   └ [저장 성공] {clean_title[:30]}...")
            except Exception as e:
                print(f"   └ [저장 실패] 중복 또는 에러: {e}")
    else:
        print(f"   └ [API 에러] {response.status_code}")

def run_all_categories():
    """정의된 8개 분야를 순회하며 수집 실행"""
    print(f"\n◈ {len(TOPICS)}개 분야 뉴스 수집 사이클 시작 ◈")
    for topic in TOPICS:
        print(f"▶ [{topic}] 분야 수집 중...")
        collect_news(topic, count=1)
    print("◈ 모든 분야 수집 프로세스 종료 ◈\n")

if __name__ == "__main__":
    # 파일 단독 실행 시 8개 분야 모두 수집
    run_all_categories()