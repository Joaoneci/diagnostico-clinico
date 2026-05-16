FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY 24_rag_medico_langgraph_ecra/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY lib_llm_fallback.py .
COPY 24_rag_medico_langgraph_ecra/ ./ 

# Run gerador_dados on start to populate DB if needed, but it's better to run manually or gracefully inside the app.
# We will just expose streamlit.
EXPOSE 8501

CMD ["sh", "-c", "python gerador_dados.py && streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"]
