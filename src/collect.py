import os
import requests
import re
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. 금고(.env) 열어서 네이버 & Supabase 열쇠 모두 꺼내기
load_dotenv()
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# 2. Supabase DB에 접속 준비
supabase: Client = create_client(supabase_url, supabase_key)

# 3. 네이버 뉴스 API 호출 (인공지능 관련 최신 뉴스 5개)
url = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": naver_client_id,
    "X-Naver-Client-Secret": naver_client_secret
}
params = {"query": "인공지능", "display": 5, "sort": "date"}

print("📡 네이버 서버에서 뉴스를 가져옵니다...")
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    insert_data = [] # DB에 넣을 뉴스들을 잠시 모아둘 장바구니
    
    for item in data['items']:
        # <b> 태그 등 특수기호 깔끔하게 지우기 (전처리)
        title = re.sub(r'<[^>]+>', '', item['title'])
        title = title.replace('&quot;', '"').replace('&apos;', "'")
        link = item['originallink']
        
        # 장바구니에 제목과 링크 담기 (DB 컬럼 이름과 똑같아야 함!)
        insert_data.append({
            "title": title,
            "originallink": link
        })
        print(f"- {title}")

    # 4. 장바구니에 담긴 뉴스를 Supabase DB로 쏘기 (Insert)
    if insert_data:
        print(f"\n💾 {len(insert_data)}개의 뉴스를 Supabase DB에 저장합니다...")
        db_response = supabase.table("news_data").insert(insert_data).execute()
        print("✅ DB 저장 완료!! Supabase 화면에서 확인해 보세요!")

else:
    print(f"❌ 에러 발생 (코드: {response.status_code})")