import os
import re
import json
import requests
import time
from dotenv import load_dotenv
from supabase import create_client
import mlflow  # 🚨 MLflow 라이브러리 연동
import litellm # 🚨 AI Gateway 역할을 수행할 LiteLLM 연동

# 환경 설정 및 초기화
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

from advanced_eval import get_comprehensive_score


def evaluate_models():
    print("\n" + "="*50)
    print("🎯 [AI 심사위원 평가 + MLOps 메트릭 로깅 시작]")
    
    response = supabase.table("news_data").select("*").is_("score_kobart", "null").order("id", desc=True).execute()
    
    if not response.data:
        print("💡 평가할 새로운 데이터가 없습니다.")
        return

    # 🚀 [해결 포인트] MLflow 초기화 및 룸 세팅 코드를 함수 내부 안전지대로 이동
    target_run_name = "Continuous_MLOps_Monitoring"
    active_run_id = None
    run_context = None
    
    try:
        # MLflow 연결 정보 주입 및 실험방 설정
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("News_Summary_MLOps_Project")
        
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
                
        # 룸 세션 오픈
        run_context = mlflow.start_run(run_id=active_run_id, run_name=target_run_name)
    except Exception as e:
        # 🚨 여기서 예외가 잡히므로 서버가 꺼져있거나 클라우드 환경이어도 아래 코드가 중단되지 않고 정상 진행됨
        print(f"⚠️ [MLflow Warning] 트래킹 서버 미가동(또는 클라우드 환경). 대시보드 전송을 우회하고 DB 적재에 집중합니다.")

    # 실질적인 평가 및 DB 적재 루프 진입
    try:
        for news in response.data:
            news_id = news['id']
            topic = news.get('topic', '미분류')
            print(f"\n📝 대상 기사 ID: {news_id} [{topic}]")
            
            update_data = {}
            for m in ["kobart", "kot5", "roberta"]:
                summary = news.get(f"summary_{m}")
                if not summary or len(summary) < 5: continue
                
                print(f" > {m.upper()} 채점 진행 중 (QAFactEval & G-Eval)...", end=" ", flush=True)
                final_score = get_comprehensive_score(summary, news['description'])
                
                if final_score > 0:
                    
                    if final_score > 100.0: final_score = 100.0
                    elif final_score < 0.0: final_score = 0.0
                        
                    update_data[f"score_{m}"] = final_score
                    print(f"✅ {final_score}점")
                    
                    # MLflow 로깅 세션이 안전하게 수립되었을 때만 메트릭 전송 시도
                    if run_context and mlflow.active_run():
                        try:
                            mlflow.log_metric(f"score_{m}_{topic}", final_score, step=int(news_id))
                        except Exception:
                            pass
                else:
                    update_data[f"score_{m}"] = 0.0
                    print("❌ 평가 불가 (0점 처리)")
                time.sleep(0.5)

            if update_data:
                supabase.table("news_data").update(update_data).eq("id", news_id).execute()

        print("\n🏆 DB 업데이트 및 인프라 파이프라인 정상 종료!")
        
    finally:
        # 어떤 상황에서든 세션 컨텍스트가 열려있었다면 깔끔하게 반환하고 종료
        if run_context:
            try:
                mlflow.end_run()
            except Exception:
                pass

if __name__ == "__main__":
    evaluate_models()