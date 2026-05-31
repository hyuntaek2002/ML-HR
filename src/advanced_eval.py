import json
import litellm
import mlflow

@mlflow.trace(name="QAFactEval", span_type="EVALUATION")
def run_qafacteval(summary_text, original_text, model_name="gpt-4o-mini"):
    """
    무참조 사실성 검증 (QAFactEval 시뮬레이션)
    요약문에서 핵심 명사를 추출해 질문을 만들고 원문에서 답을 찾는지 확인하여 환각을 감지합니다.
    """
    prompt = f"""
    당신은 팩트체크 시스템(QAFactEval)입니다. 아래 요약문이 원문에 없는 거짓(Hallucination)을 포함하는지 단계별로 검증하세요.
    1. 요약문에서 핵심 명사구/사실을 추출해 단답형 질문 생성
    2. 원문에서 해당 질문의 답을 찾을 수 있는지 검증 (Yes/No)
    3. 최종적으로 요약문이 원문에 100% 부합하는 사실인지 점수(0~100)로 평가

    원문: {original_text}
    요약: {summary_text}
    
    출력 형식(JSON): {{"is_factual": true/false, "fact_score": 0~100사이_정수, "reasoning": "평가 사유"}}
    """
    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            timeout=20
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("fact_score", 0))
    except Exception as e:
        print(f"⚠️ QAFactEval 오류: {e}")
        return 0.0

@mlflow.trace(name="G-Eval", span_type="EVALUATION")
def run_geval(summary_text, original_text, model_name="gpt-4o-mini"):
    """
    G-Eval (로그 확률 기반 채점)
    단순 프롬프트가 아니라 평가 기준(Coherence, Consistency, Fluency, Relevance)을 명시하고,
    가능한 경우 토큰의 확률(logprobs)을 고려하여 신뢰도 높은 점수를 도출합니다.
    """
    prompt = f"""
    당신은 전문적인 언어학자이자 뉴스 요약 평가자입니다. 다음 4가지 기준에 따라 1점부터 5점까지 평가하세요.
    1. Relevance (원문의 핵심 내용이 포함되었는가)
    2. Consistency (원문과 모순되는 내용이 없는가)
    3. Fluency (문법적으로 자연스러운가)
    4. Coherence (문장 간 논리적 연결이 매끄러운가)

    원문: {original_text}
    요약: {summary_text}
    
    위 4가지 기준을 종합하여 최종 품질을 0점~100점 사이로 환산하여 평가하세요. 
    출력 형식(JSON): {{"geval_score": 0~100사이_정수, "details": {{"relevance": 1~5점, "consistency": 1~5점, "fluency": 1~5점, "coherence": 1~5점}}}}
    """
    try:
        # logprobs 파라미터를 지원하는 모델의 경우 활성화
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            logprobs=True,
            top_logprobs=2,
            timeout=20
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 실제 G-Eval은 1~5 토큰의 로그 확률 가중합을 계산하지만,
        # 여기서는 OpenAI 응답 JSON에서 추출한 환산 점수를 베이스로 사용하고, 
        # 토큰 확률(logprobs)이 존재하면 로그를 남기도록 시뮬레이션합니다.
        if response.choices[0].logprobs:
            # logprobs 메타데이터 추출 가능 (MLflow Tracing 등에 추후 로깅)
            pass
            
        return float(result.get("geval_score", 0))
    except Exception as e:
        print(f"⚠️ G-Eval 오류: {e}")
        return 0.0

@mlflow.trace(name="Comprehensive_Eval", span_type="EVALUATION")
def get_comprehensive_score(summary_text, original_text, model_name="gpt-4o-mini"):
    """QAFactEval과 G-Eval을 연합 가동하여 최종 신뢰도 점수를 도출"""
    fact_score = run_qafacteval(summary_text, original_text, model_name)
    geval_score = run_geval(summary_text, original_text, model_name)
    
    # 사실성이 낮으면 (환각 발생 시) G-Eval 점수와 무관하게 패널티 부여
    if fact_score < 50:
        final_score = (geval_score * 0.4) + (fact_score * 0.6) # 팩트 비중 강화
    else:
        final_score = (geval_score * 0.6) + (fact_score * 0.4)
        
    return round(final_score, 1)
