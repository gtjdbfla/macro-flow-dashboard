FROM python:3.11-slim

ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y --no-install-recommends tzdata curl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 폴더를 하나씩 적다 빠뜨려 배포가 두 번 깨졌다(scoring/). 통째로 넣고 .dockerignore로 거른다.
COPY . .

EXPOSE 8503
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8503/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
