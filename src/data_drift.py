import os
import pandas as pd
import textstat
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from supabase import create_client
from dotenv import load_dotenv

# 환경 설정
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def compute_text_features(text):
    """텍스트의 가독성(Flesch Reading Ease)을 계산합니다."""
    # textstat은 기본적으로 영어에 최적화되어 있으나, 
    # 한국어 길이나 구조 변화(문장당 단어 수 등)를 수치화하는 데에도 통계적(Proxy)으로 유용합니다.
    try:
        if not text or len(text.strip()) == 0:
            return 0.0
        return textstat.flesch_reading_ease(text)
    except:
        return 0.0

def detect_drift_for_topic(topic, current_texts):
    """
    Supabase에서 최근 과거 데이터를 준거(Reference) 데이터로 가져와,
    현재 들어온 텍스트(Current)와의 PSI(인구 안정성 지수) 기반 데이터 드리프트를 계산합니다.
    """
    # 1. 준거(Reference) 데이터 로드 (최근 요약 완료된 50건)
    res = supabase.table("news_data").select("description")\
        .eq("topic", topic).not_.is_("score_kobart", "null").order("id", desc=True).limit(50).execute()
        
    if not res.data or len(res.data) < 10:
        # 과거 데이터가 너무 적으면 판단 보류
        return False, 0.0
        
    ref_texts = [d['description'] for d in res.data if d.get('description')]
    
    # 2. DataFrame 구성 및 특징(Feature) 추출
    df_ref = pd.DataFrame({"text": ref_texts})
    df_cur = pd.DataFrame({"text": current_texts})
    
    df_ref["readability"] = df_ref["text"].apply(compute_text_features)
    df_cur["readability"] = df_cur["text"].apply(compute_text_features)
    
    # 3. Evidently AI를 이용한 PSI 기반 통계적 드리프트 감지 (임계값 0.25)
    report = Report(metrics=[
        ColumnDriftMetric(column_name="readability", stattest="psi", stattest_threshold=0.25)
    ])
    
    # 데이터가 너무 적을 때 Evidently가 에러를 뱉을 수 있으므로 예외 처리
    try:
        report.run(reference_data=df_ref, current_data=df_cur)
        results = report.as_dict()
        
        drift_result = results["metrics"][0]["result"]
        drift_detected = drift_result["drift_detected"]
        drift_score = drift_result["drift_score"]
        
        return drift_detected, drift_score
    except Exception as e:
        print(f"⚠️ Evidently PSI 계산 중 예외 발생: {e}")
        return False, 0.0

def isolate_drifted_data(topic, texts, psi_score):
    """드리프트가 감지된 텍스트를 격리 처리(DB 또는 별도 스토리지)합니다."""
    print(f"🚨 [경보] {topic} 분야 데이터 드리프트 감지! (PSI: {psi_score:.3f})")
    print("   └> 해당 데이터는 요약 파이프라인에서 제외하고 lakeFS(격리 스토리지)로 이동됩니다.")
    # 실제 프로덕션에서는 여기서 lakeFS 분기를 타거나 별도 DB 테이블에 넣음
    for text in texts:
        supabase.table("drift_isolated_data").insert({
            "topic": topic,
            "description": text,
            "psi_score": psi_score
        }).execute()
