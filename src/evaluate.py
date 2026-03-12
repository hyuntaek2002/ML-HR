import time
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from preprocess import clean_text 

# 1. MLflow 세팅
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("News_Summary_Evaluation")

# 2. 한국어 요약의 대명사 (가장 안정적인 모델)
print("🤖 가장 대중적인 KoBART 모델을 가져오는 중입니다...")
model_name = "digit82/kobart-summarization"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except Exception as e:
    print(f"❌ 모델 로드 중 에러 발생: {e}")
    print("💡 네트워크 문제일 수 있습니다. 잠시 후 다시 시도하거나 인터넷 연결을 확인해 주세요.")
    exit()

# 3. 데이터 준비(test)
raw_news = """
인공지능(AI) 기술이 빠르게 발전하면서 산업 전반에 걸쳐 혁신이 일어나고 있다. 특히 생성형 AI는 텍스트, 이미지, 영상 등 다양한 형태의 콘텐츠를 인간과 유사한 수준으로 만들어내며 큰 주목을 받고 있다. 전문가들은 이러한 AI 기술이 업무 효율성을 극대화하고 새로운 비즈니스 모델을 창출할 것으로 기대하고 있다. 하지만 일각에서는 AI가 인간의 일자리를 대체할 수 있다는 우려의 목소리도 나오고 있으며, 저작권 침해나 가짜 뉴스 생성과 같은 윤리적 문제에 대한 해결책 마련이 시급하다는 지적도 제기된다. 이에 따라 정부와 기업들은 AI 기술 발전과 함께 이를 규제하고 관리할 수 있는 법적, 제도적 가이드라인을 준비하는 데 집중하고 있다.
"""
reference_summary = "생성형 AI 기술의 발전으로 산업 혁신과 업무 효율성 증대가 기대되지만, 일자리 감소와 윤리적 문제에 대한 우려로 인해 법적, 제도적 가이드라인 마련이 시급한 상황이다."

clean_news = clean_text(raw_news)

# 4. 모델 평가 실행
with mlflow.start_run(run_name="KoBART_Digit82_Test"):
    mlflow.log_param("model_name", model_name)
    
    start_time = time.time()
    
    # 입력 인코딩
    inputs = tokenizer(clean_news, return_tensors="pt", truncation=True, max_length=512)
    
    # 요약 생성
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=100,
        min_length=20,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    ai_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    latency = time.time() - start_time
    
    # ROUGE 점수 채점
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, ai_summary)
    rouge_l_score = scores['rougeL'].fmeasure * 100
    
    mlflow.log_metric("Latency", latency)
    mlflow.log_metric("ROUGE_L", rouge_l_score)
    
    print("\n" + "="*60)
    print(f"🤖 [AI 요약 결과]: {ai_summary}")
    print("="*60)
    print(f"📊 [성적표] 시간: {latency:.2f}초 / ROUGE: {rouge_l_score:.1f}점")