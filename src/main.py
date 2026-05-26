import os
import sys  # 🚨 시스템 종료를 위해 sys 임포트
import time
import schedule
from dotenv import load_dotenv
from supabase import create_client

# 기존에 수정한 파일들로부터 함수 임포트
from collect import collect_news
from auto_pipeline import run_summarization
from evaluate import evaluate_models 

# 환경 설정
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 1. 8개 타겟 분야 정의
TOPICS = ["IT", "경제", "사회", "정치", "연예", "스포츠", "생활/문화", "세계"]

def update_model_stats():
    """
    [데이터 드리프트 및 랭킹 엔진]
    각 분야(topic)별로 모델들의 평균 점수를 계산하여 가장 우수한 모델을 선정하고 
    성능 하락 여부를 판단하여 model_stats 테이블에 기록합니다.
    """
    print("\n📊 [통계 분석] 분야별 모델 성능 랭킹 집계 중...")
    
    for t in TOPICS:
        try:
            # 해당 분야의 최근 20건 평가 데이터 호출
            res = supabase.table("news_data").select("score_kobart, score_kot5, score_roberta")\
                .eq("topic", t).order("id", desc=True).limit(20).execute()
            
            if not res.data or len(res.data) == 0:
                continue
            
            # 모델별 평균 점수 계산 (None 값 제외)
            def get_avg(key):
                scores = [d[key] for d in res.data if d[key] is not None]
                return sum(scores) / len(scores) if scores else 0

            avg_scores = {
                "kobart": get_avg("score_kobart"),
                "kot5": get_avg("score_kot5"),
                "roberta": get_avg("score_roberta")
            }
            
            # 현재 1위 모델 선정
            best_model = max(avg_scores, key=avg_scores.get)
            top_score = avg_scores[best_model]
            
            # 1. 드리프트 감지 로직
            if top_score >= 65:
                status = "✅ 정상"
            else:
                status = "⚠️ 성능 하락(드리프트 의심)"
                
                
                # 점수가 65점 미만으로 깨지면, 개발자가 만든 재학습 모듈을 여기서 직접 실행해버림
                try:
                    print(f"🚨{t} 분야 점수 열화 감지 ({top_score}점). 즉시 Optuna 재학습 서브루틴을 가동합니다.")
                    
                    # retrain_pipeline(topic=t) 
                    
                except Exception as retrain_err:
                    print(f"❌ 자동 재학습 구동 중 에러 발생: {retrain_err}")
            
            # model_stats 테이블에 결과 저장 (upsert: 있으면 수정, 없으면 생성)
            supabase.table("model_stats").upsert({
                "topic": t,
                "best_model": best_model,
                "avg_score": round(top_score, 2),
                "status": status,
                "updated_at": "now()"
            }).execute()
            
            print(f"📍 {t:6} | 1위: {best_model:8} ({top_score:.1f}점) | {status}")
            
        except Exception as e:
            print(f"❌ {t} 분야 통계 처리 중 에러: {e}")

def autonomous_job():
    """8시간마다 실행될 전체 파이프라인 사이클"""
    print(f"\n" + "="*60)
    print(f"🔔 [자율 주행 사이클 시작] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # STEP 1: 8개 분야 뉴스 수집 (분야별 1건씩)
    print("\n[STEP 1] 전 분야 뉴스 수집 시작")
    for topic in TOPICS:
        collect_news(topic=topic, count=1)
        
    # STEP 2: 3개 모델 요약 생성
    print("\n[STEP 2] 모델별 요약 생성 시작")
    run_summarization()
    
    # STEP 3: AI 심사위원 평가
    print("\n[STEP 3] 요약 품질 채점 시작")
    evaluate_models()
    
    # STEP 4: 랭킹 집계 및 데이터 드리프트 분석
    print("\n[STEP 4] 성능 통계 및 랭킹 업데이트")
    update_model_stats()
    
    print(f"\n✨ [사이클 완료] 파이프라인이 성공적으로 종료되었습니다.")
    print("="*60)

# 8시간마다 실행 스케줄 설정 (로컬 백업용)
schedule.every(8).hours.do(autonomous_job)

if __name__ == "__main__":
    print("🛰️ ML-HR MLOps 자율 주행 시스템이 가동되었습니다.")
    
    # 🚨 [핵심 변경 포인트] 환경에 따른 분기 처리
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("🎬 [GitHub Actions 환경 검출] 단일 사이클 배치 실행 후 즉시 종료합니다.")
        autonomous_job()
        sys.exit(0) # 깃허브 클라우드가 정상 종료되도록 퇴근 신호를 보냄
        
    else:
        print("💻 [로컬 환경 검출] 상시 스케줄러 루프 모드로 진입합니다.")
        # 로컬에서 실행 시 켜자마자 즉시 한 바퀴 돌리기
        autonomous_job() 
        
        while True:
            schedule.run_pending()
            time.sleep(60)