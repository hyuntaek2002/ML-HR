import os
import re
import json
import openai
import requests
import time
from dotenv import load_dotenv
from supabase import create_client
import mlflow  # 🚨 MLflow 라이브러리 추가

# 환경 설정 및 초기화
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🚨 MLflow 트래킹 서버 주소 설정 (로컬 기본값: http://localhost:5000)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("News_Summary_MLOps_Project")  # 교수님이 보실 실험 이름

def get_gpt_score(summary_text, original_text):
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
    url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    headers = {
        "Authorization": f"Bearer {os.getenv('CLOVA_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": f"뉴스 원문과 요약문을 비교하여 요약 품질을 0~100 사이의 숫자로만 평가해줘.\n\n원문: {original_text}\n요약: {summary_text}"}],
        "topP": 0.8, "topK": 0, "maxTokens": 256, "temperature": 0.1, "stopBefore": [], "repeatPenalty": 1.1
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
        print(f"\n    ❌ Clova 호출 중 예외 발생: {e}")
        return None

def evaluate_models():
    print("\n" + "="*50)
    print("🎯 [AI 심사위원 평가 + MLflow 로깅 시작]")
    
    response = supabase.table("news_data").select("*").is_("score_kobart", "null").order("id", desc=True).execute()
    
    if not response.data:
        print("💡 평가할 새로운 데이터가 없습니다.")
        return

    # 🚨 [수정 포인트] run_name을 고정하거나 open된 active run을 재사용합니다.
    # 이렇게 하면 하나의 RunID 안에 모든 배치의 데이터가 Step(뉴스ID)별로 축적됩니다.
    with mlflow.start_run(run_name="Continuous_MLOps_Monitoring", nested=True):
        
        for news in response.data:
            news_id = news['id']
            topic = news.get('topic', '미분류')
            print(f"\n📝 대상 기사 ID: {news_id} [{topic}]")
            
            update_data = {}
            for m in ["kobart", "kot5", "roberta"]:
                summary = news.get(f"summary_{m}")
                if not summary or len(summary) < 5: continue
                
                print(f" > {m.upper()} 채점 진행 중...", end=" ", flush=True)
                s1 = get_gpt_score(summary, news['description'])
                s2 = get_clova_score(summary, news['description'])
                
                valid_scores = [s for s in [s1, s2] if s is not None]
                if valid_scores:
                    final_score = round(sum(valid_scores) / len(valid_scores), 1)
                    
                    if final_score > 100.0: final_score = 100.0
                    elif final_score < 0.0: final_score = 0.0
                        
                    update_data[f"score_{m}"] = final_score
                    print(f"✅ {final_score}점")
                    
                    # 🚨 step 인자에 뉴스 ID(데이터 고유번호)를 넣어주어 그래프의 X축이 우측으로 전진하게 합니다.
                    mlflow.log_metric(f"score_{m}_{topic}", final_score, step=int(news_id))
                else:
                    update_data[f"score_{m}"] = 0.0
                    print("❌ 평가 불가")
                time.sleep(0.5)

            if update_data:
                supabase.table("news_data").update(update_data).eq("id", news_id).execute()

        print("\n🏆 DB 업데이트 및 MLflow 트래킹 대시보드 전송 완료!")

if __name__ == "__main__":
    evaluate_models()