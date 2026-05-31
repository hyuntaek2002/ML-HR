import os
import gc
import math
import re
import numpy as np
import torch
import pandas as pd
import textstat
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
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
        return float(textstat.flesch_reading_ease(text))
    except:
        return 0.0

def compute_burstiness(text):
    """
    텍스트 내부 문장 길이(단어 수)의 변동계수(Burstiness)를 계산합니다.
    표준편차 / 평균으로 산출하며, 길이가 고정적이면 낮고 다채로우면 높습니다.
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # 문장 단위로 거칠게 분리 (., !, ? 기준)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    if len(sentences) == 0:
        return 0.0
        
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    
    if mean_len == 0:
        return 0.0
        
    std_len = np.std(lengths)
    burstiness = std_len / mean_len
    return float(burstiness)

def compute_perplexity(text_series):
    """
    Pandas Series 형태의 텍스트 리스트를 받아 KoGPT2로 퍼플렉서티를 일괄 계산합니다.
    메모리 최적화를 위해 호출 시에만 모델을 로드하고 연산 후 즉시 삭제합니다.
    """
    model_name = 'skt/kogpt2-base-v2'
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
        
        perplexities = []
        for text in text_series:
            if not text or len(text.strip()) == 0:
                perplexities.append(0.0)
                continue
            
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            if inputs['input_ids'].size(1) == 0:
                perplexities.append(0.0)
                continue
                
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                # loss가 너무 크면 math.exp 오버플로우가 날 수 있으므로 방어
                try:
                    ppl = math.exp(loss.item())
                except OverflowError:
                    ppl = 10000.0 # 상한치 캡
            perplexities.append(float(ppl))
            
    finally:
        # 모델 완전 삭제 및 가비지 컬렉션 (메모리 누수 방지)
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    return perplexities

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
    
    df_ref["burstiness"] = df_ref["text"].apply(compute_burstiness)
    df_cur["burstiness"] = df_cur["text"].apply(compute_burstiness)
    
    print("   └> [Drift] 퍼플렉서티 연산용 언어 모델을 임시 로드합니다...")
    df_ref["perplexity"] = compute_perplexity(df_ref["text"])
    df_cur["perplexity"] = compute_perplexity(df_cur["text"])
    
    # 3. Evidently AI를 이용한 3차원 다변량 드리프트 감지 (임계값 0.25)
    report = Report(metrics=[
        ColumnDriftMetric(column_name="readability", stattest="psi", stattest_threshold=0.25),
        ColumnDriftMetric(column_name="burstiness", stattest="psi", stattest_threshold=0.25),
        ColumnDriftMetric(column_name="perplexity", stattest="psi", stattest_threshold=0.25)
    ])
    
    # 데이터가 너무 적을 때 Evidently가 에러를 뱉을 수 있으므로 예외 처리
    try:
        report.run(reference_data=df_ref, current_data=df_cur)
        results = report.as_dict()
        
        drifted_features = 0
        total_drift_score = 0.0
        
        for metric in results["metrics"]:
            res = metric["result"]
            score = res["drift_score"]
            total_drift_score += score
            if res["drift_detected"]:
                drifted_features += 1
                
        avg_drift_score = total_drift_score / 3.0
        
        # 3개 중 2개 이상 지표가 박살났거나 평균 분포가 심하게 벗어나면 드리프트 판정
        drift_detected = (drifted_features >= 2) or (avg_drift_score > 0.25)
        
        return drift_detected, avg_drift_score
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
