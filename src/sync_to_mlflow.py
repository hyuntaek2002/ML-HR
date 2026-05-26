import os
import sys
from dotenv import load_dotenv
from supabase import create_client
import mlflow

# 파이썬 경로 인식 버그 방지
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# MLflow 주소 및 실험 설정
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("News_Summary_MLOps_Project")

def backfill_past_data():
    print("🔄 [Backfill] Supabase DB에서 과거 채점 완료 데이터를 추출 중...")
    
    # 이미 채점 완료된(score_kobart가 null이 아닌) 데이터를 ID 순서대로 가져옵니다.
    res = supabase.table("news_data").select("*").not_.is_("score_kobart", "null").order("id").execute()
    
    if not res.data:
        print("💡 동기화할 과거 데이터가 없습니다.")
        return
        
    print(f"📊 총 {len(res.data)}건의 기사 실적을 발견했습니다. MLflow 연속 그래프로 이식을 시작합니다. (AI 호출 0건)")
    
    # 우리가 수정한 '하나의 연속된 일기장(Run)'을 엽니다.
    with mlflow.start_run(run_name="Continuous_MLOps_Monitoring"):
        for news in res.data:
            news_id = news['id']
            topic = news.get('topic', '미분류')
            
            # 3개 모델의 기존 점수를 읽어서 MLflow에 스텝별로 주입
            for m in ["kobart", "kot5", "roberta"]:
                score = news.get(f"score_{m}")
                if score is not None:
                    # 🚨 핵심: AI를 호출하지 않고, 오직 저장된 수치만 MLflow 타임라인에 매핑합니다.
                    mlflow.log_metric(f"score_{m}_{topic}", score, step=int(news_id))
                    
    print("\n✅ [동기화 최종 완료] 과거의 모든 데이터가 단 1원의 추가 지출 없이 MLflow 그래프로 통합되었습니다!")

if __name__ == "__main__":
    backfill_past_data()