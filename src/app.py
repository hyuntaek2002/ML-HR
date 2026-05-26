import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client

# 환경 변수 및 Supabase 연결
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 1. 웹페이지 기본 설정 (와이드 레이아웃)
st.set_page_config(
    page_title="ML-HR MLOps Dashboard", 
    page_icon="🛰️", 
    layout="wide"
)

# 대시보드 메인 타이틀
st.title("🛰️ ML-HR 요약 및 모니터링 시스템")
st.markdown("8시간마다 8개 분야의 뉴스를 수집하고, **BART / T5 / RoBERTa** 모델의 요약 품질을 실시간 감시하는 **MLOps 대시보드**입니다.")
st.write("---")

# 2. 상단 탭 구성 (랭킹 화면 / 상세 비교 화면)
tab1, tab2 = st.tabs(["📊 분야별 랭킹 & 드리프트 모니터링", "📝 최신 요약 비교 피드"])

# ==========================================
# TAB 1: 분야별 랭킹 & 데이터 드리프트 현황
# ==========================================
with tab1:
    st.header("🏆 분야(Topic)별 최적 모델 성적표")
    st.caption("최근 20건의 AI 심사위원(GPT/Clova) 점수를 기반으로 산출된 실시간 순위입니다.")
    
    try:
        # DB에서 통계 데이터 읽어오기
        res_stats = supabase.table("model_stats").select("*").order("topic").execute()
        
        if res_stats.data and len(res_stats.data) > 0:
            # Pandas 데이터프레임으로 변환하여 예쁘게 가공
            df = pd.DataFrame(res_stats.data)
            
            # 사용자 정렬 및 컬럼명 가독성 개선
            df = df[['topic', 'best_model', 'avg_score', 'status', 'updated_at']]
            df.columns = ['분야 (Topic)', '현재 1위 모델 (Champion)', '최근 20건 평균 점수', '모니터링 상태', '최종 갱신 시간']
            
            # 화면에 표 출력
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 드리프트 발생 구역 요약 띄워주기
            drift_count = sum(1 for d in res_stats.data if "드리프트" in d.get('status', ''))
            if drift_count > 0:
                st.error(f"⚠️ 현재 8개 분야 중 {drift_count}개 분야에서 성능 저하(데이터 드리프트)가 감지되어 모델 재학습 검토가 필요합니다.")
            else:
                st.success("✅ 모든 분야의 모델이 안정적인 품질(65점 이상)을 유지하며 정상 가동 중입니다.")
                
        else:
            st.info("💡 아직 누적된 통계 데이터가 없습니다. `main.py` 파이프라인이 최소 1회 완전히 종료되어야 기록됩니다.")
    except Exception as e:
        st.error(f"통계 데이터를 불러오는 중 오류 발생: {e}")

# ==========================================
# TAB 2: 모델별 요약문 진검승부 비교 피드
# ==========================================
with tab2:
    st.header("📰 최신 수집 뉴스 및 모델별 요약 비교")
    st.caption("가장 최근에 수집된 뉴스 8건을 대상으로 세 모델의 출력물과 점수를 직접 비교합니다.")
    
    try:
        # 최근 뉴스 8건 가져오기
        res_news = supabase.table("news_data").select("*").order("id", desc=True).limit(8).execute()
        
        if res_news.data:
            for news in res_news.data:
                topic = news.get('topic', '미분류')
                title = news.get('title', '제목 없음')
                
                # 아코디언 메뉴(Expander) 형태로 접고 펼 수 있게 구성
                with st.expander(f"📌 [{topic}] {title}"):
                    st.write("**[뉴스 원문 기사 본문]**")
                    st.caption(news.get('description', '본문 내용이 없습니다.'))
                    
                    st.write("---")
                    st.write("**🤖 모델별 요약 성능 대조**")
                    
                    # 화면을 가로로 3분할 (BART / T5 / RoBERTa)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🟢 KoBART")
                        score = news.get('score_kobart')
                        st.metric(label="심사 점수", value=f"{score} 점" if score is not None else "채점 대기")
                        st.info(news.get('summary_kobart', '요약문 생성 전입니다.'))
                        
                    with col2:
                        st.markdown("### 🔵 T5")
                        score = news.get('score_kot5')
                        st.metric(label="심사 점수", value=f"{score} 점" if score is not None else "채점 대기")
                        st.success(news.get('summary_kot5', '요약문 생성 전입니다.'))
                        
                    with col3:
                        st.markdown("### 🟡 RoBERTa")
                        score = news.get('score_roberta')
                        st.metric(label="심사 점수", value=f"{score} 점" if score is not None else "채점 대기")
                        st.warning(news.get('summary_roberta', '요약문 생성 전입니다.'))
        else:
            st.info("💡 요약된 뉴스 데이터가 아직 없습니다. 파이프라인을 가동해 주세요.")
    except Exception as e:
        st.error(f"뉴스 데이터를 불러오는 중 오류 발생: {e}")