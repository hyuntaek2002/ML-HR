# Python 3.10 이상의 런타임 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 코드 복사
COPY . .

# FastAPI 포트 개방
EXPOSE 8000
# Streamlit 포트 개방
EXPOSE 8501
# MLflow 포트 개방
EXPOSE 5000

# 기본 실행 커맨드는 docker-compose에서 덮어씁니다.
CMD ["python", "src/main.py"]
