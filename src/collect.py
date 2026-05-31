import os
import html  # HTML 특수문자 완벽 세탁용 라이브러리 추가
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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
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
    """특정 분야(topic)의 뉴스를 수집하여 DB에 저장 (Drift 감지 포함)"""
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    
    # [안전장치] URL 한글 깨짐 방지를 위해 params 딕셔너리 구조로 변경
    base_url = "https://openapi.naver.com/v1/search/news.json"
    params = {
        "query": topic,
        "display": count,
        "sort": "sim"
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            items = response.json().get('items', [])
            
            # 수집된 텍스트 리스트 만들기
            collected_data = []
            current_texts = []
            
            for item in items:
                full_text = get_full_text(item['link'])
                if not full_text or len(full_text) < 50:
                    full_text = item['description']
                
                # [세탁] b 태그뿐만 아니라 모든 HTML 특수문자(&amp; 등)를 한 번에 한글로 디코딩
                clean_title = html.unescape(item['title']).replace('<b>', '').replace('</b>', '')
                clean_desc = html.unescape(full_text).replace('<b>', '').replace('</b>', '')
                
                data = {
                    "title": clean_title,
                    "description": clean_desc, 
                    "originallink": item['originallink'],
                    "topic": topic
                }
                collected_data.append(data)
                current_texts.append(clean_desc)
                
            if current_texts:
                from data_drift import detect_drift_for_topic, isolate_drifted_data
                
                # 데이터 드리프트(PSI 등) 실시간 확인
                drift_detected, psi_score = detect_drift_for_topic(topic, current_texts)
                
                if drift_detected and psi_score > 0.25:
                    # 격리 처리 (news_data 테이블에 넣지 않음)
                    isolate_drifted_data(topic, current_texts, psi_score)
                else:
                    # 정상 데이터만 DB 적재 및 LakeFS 스냅샷 보존
                    for data in collected_data:
                        try:
                            supabase.table("news_data").insert(data).execute()
                            # 데이터 레이크(lakeFS) 스냅샷용 별도 로깅
                            supabase.table("lakefs_snapshot").insert({
                                "original_data": data,
                                "snapshot_timestamp": "now()"
                            }).execute()
                            print(f"   └ [저장 성공] {data['title'][:30]}...")
                        except Exception as e:
                            print(f"   └ [저장 실패] 중복 또는 에러: {e}")
        else:
            print(f"   └ [API 에러] {response.status_code}")
    except Exception as e:
        print(f"   └ [네트워크 에러] Naver API 연결 실패: {e}")

def run_all_categories():
    """정의된 8개 분야를 순회하며 수집 실행"""
    print(f"\n◈ {len(TOPICS)}개 분야 뉴스 수집 사이클 시작 ◈")
    for topic in TOPICS:
        print(f"▶ [{topic}] 분야 수집 중...")
        collect_news(topic, count=1)
    print("◈ 모든 분야 수집 프로세스 종료 ◈\n")

if __name__ == "__main__":
    run_all_categories()