import os
import re
import json
import openai
import requests
import time
from dotenv import load_dotenv
from supabase import create_client

# 환경 설정 및 API 클라이언트 초기화
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_score(summary_text, original_text):
    """GPT-4o-mini를 이용한 요약 품질 평가"""
    prompt = f"원문: {original_text}\n요약: {summary_text}\n품질을 0~100점 사이 JSON으로 평가해: {{\"score\": 0}}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0))
    except Exception as e:
        print(f"\n    ❌ GPT 평가 실패: {e}")
        return None

def get_clova_score(summary_text, original_text):
    """HyperCLOVA X HCX-005 V3를 이용한 요약 품질 평가"""
    url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    headers = {
        "Authorization": f"Bearer {os.getenv('CLOVA_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"뉴스 원문과 요약문을 비교하여 요약 품질을 0~100 사이의 숫자로만 평가해줘.\n\n원문: {original_text}\n요약: {summary_text}"
            }
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 256,
        "temperature": 0.1,
        "stopBefore": [],
        "repeatPenalty": 1.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['result']['message']['content']
            score_match = re.search(r'\d+(\.\d+)?', content)
            if score_match:
                return float(score_match.group())
        return None
    except Exception as e:
        print(f"\n    ❌ Clova 호출 예외: {e}")
        return None

def evaluate_models():
    """DB에서 요약은 완료되었으나 평가(score)가 없는 모든 뉴스를 채점합니다."""
    print("\n" + "="*50)
    print("⚖️ [AI 심사위원 평가 사이클 시작]")
    
    # 1. 평가 전(score_kobart가 NULL)인 모든 데이터 호출 (전 분야 대응)
    response = supabase.table("news_data").select("*").is_("score_kobart", "null").execute()
    news_list = response.data
    
    if not news_list:
        print("💡 현재 평가할 새로운 뉴스가 없습니다.")
        return

    print(f"🧐 총 {len(news_list)}건의 요약문을 심사합니다.")

    for news in news_list:
        news_id = news['id']
        news_topic = news.get('topic', '미분류')
        print(f"\n[평가 진행] ID: {news_id} | 분야: {news_topic} | 제목: {news['title'][:20]}...")
        
        update_data = {}
        # 3개 모델 각각에 대해 심사 진행
        for m in ["kobart", "kot5", "roberta"]:
            summary = news.get(f"summary_{m}")
            # 요약문이 없거나 실패 메시지인 경우 건너뜀
            if not summary or "실패" in summary or len(summary) < 5:
                update_data[f"score_{m}"] = 0.0
                continue
            
            print(f"   > {m.upper()} 채점 중...", end=" ", flush=True)
            s1 = get_gpt_score(summary, news['description'])
            s2 = get_clova_score(summary, news['description'])
            
            valid_scores = [s for s in [s1, s2] if s is not None]
            if valid_scores:
                final_score = round(sum(valid_scores) / len(valid_scores), 1)
                update_data[f"score_{m}"] = final_score
                print(f"✅ {final_score}점")
            else:
                update_data[f"score_{m}"] = 0.0
                print("❌ 평가 실패(0점 처리)")
            
            time.sleep(0.5) # API 속도 제한 방지

        # 채점 결과 DB 업데이트
        if update_data:
            supabase.table("news_data").update(update_data).eq("id", news_id).execute()
            print(f"   └ ID {news_id} 점수 저장 완료")

    print("\n🏆 모든 대상에 대한 평가가 최종 완료되었습니다!")

if __name__ == "__main__":
    evaluate_models()