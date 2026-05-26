import os
import re
import json
import openai
import requests
import time
from dotenv import load_dotenv
from supabase import create_client
import mlflow  # 🚨 MLflow 라이브러리 연동

# 환경 설정 및 초기화
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MLflow 트래킹 서버 주소 설정
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("News_Summary_MLOps_Project")

def get_gpt_score(summary_text, original_text):
    """GPT-4o-mini 심사위원을 통한 요약 품질 채점"""
    prompt = f"원문: {original_text}\n요약: {summary_text}\n품질을 0~100점 사이 JSON으로 평가해: {{\"score\": 0}}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            timeout=15
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0))
    except Exception as e:
        print(f"\n    ❌ GPT 평가 실패: {e}")
        return None

def get_clova_score(summary_text, original_text):
    """Clova HyperCLOVA X 심사위원을 통한 요약 품질 채점"""
    url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
    headers = {
        "Authorization": f"Bearer {os.getenv('CLOVA_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": f"뉴스 원문 and 요약문을 비교하여 요약 품질을 0~100 사이의 숫자로만 평가해줘.\n\n원문: {original_text}\n요약: {summary_text}"}],
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
    print("🎯 [AI 심사위원 평가 + MLOps 메트릭 로깅 시작]")
    
    response = supabase.table("news_data").select("*").is_("score_kobart", "null").order("id", desc=True).execute()
    
    if not response.data:
        print("💡 평가할 새로운 데이터가 없습니다.")
        return

    # 🚀 [MLOps 기법] 연속적인 대시보드 구성을 위해 기존 Run ID 추적 및 재사용 로직
    target_run_name = "Continuous_MLOps_Monitoring"
    active_run_id = None
    
    try:
        # 기존에 생성된 동일한 이름의 Run이 있는지 조회
        current_experiment = mlflow.get_experiment_by_name("News_Summary_MLOps_Project")
        if current_experiment:
            existing_runs = mlflow.search_runs(
                experiment_ids=[current_experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{target_run_name}'",
                max_results=1
            )
            if not existing_runs.empty:
                active_run_id = existing_runs.iloc[0]["run_id"]
                print(f"🔗 [MLflow 연결] 기존 모니터링 룸(Run ID: {active_run_id})을 재사용하여 그래프를 이어 그립니다.")
    except Exception as e:
        print(f"ℹ️ MLflow 기존 룸 조회 불가 (로컬 서버가 꺼져있거나 클라우드 환경임): {e}")

    # 🚀 [Non-Blocking 가드] MLflow 서버가 안 켜져 있어도 전체 파이프라인이 정상 작동하도록 설계
    try:
        run_context = mlflow.start_run(run_id=active_run_id, run_name=target_run_name)
    except Exception as e:
        print(f"⚠️ [MLflow Warning] 트래킹 서버 연결 실패. 메트릭 로깅을 건너뛰고 DB 적재에 집중합니다.")
        run_context = None

    # 실질적인 평가 루프 진입
    try:
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
                    
                    # 이상치 클리핑 가드 가동
                    if final_score > 100.0: final_score = 100.0
                    elif final_score < 0.0: final_score = 0.0
                        
                    update_data[f"score_{m}"] = final_score
                    print(f"✅ {final_score}점")
                    
                    # MLflow 로깅 세션이 안전하게 열려있을 때만 메트릭 전송
                    if run_context and mlflow.active_run():
                        try:
                            mlflow.log_metric(f"score_{m}_{topic}", final_score, step=int(news_id))
                        except Exception as log_err:
                            pass
                else:
                    update_data[f"score_{m}"] = 0.0
                    print("❌ 평가 불가 (0점 처리)")
                time.sleep(0.5)

            if update_data:
                supabase.table("news_data").update(update_data).eq("id", news_id).execute()

        print("\n🏆 DB 업데이트 및 인프라 파이프라인 정상 종료!")
        
    finally:
        # 어떤 예외가 나더라도 오픈된 MLflow 세션은 클린하게 닫아줌
        if run_context:
            mlflow.end_run()

if __name__ == "__main__":
    evaluate_models()