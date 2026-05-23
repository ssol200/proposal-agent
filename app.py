import streamlit as st
import os
import json
import tempfile
import pandas as pd
from datetime import datetime
from docx import Document
from supabase import create_client, Client

# LlamaIndex 관련 (통합 활용)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 웹 페이지 설정
st.set_page_config(page_title="RFP 제안서 분석 & 생성 에이전트", layout="wide")

# 1. 시크릿 관리 (Streamlit Cloud 환경 맞춤 보완)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    # 기본 연결 주소를 가져옵니다.
    raw_db_connection = st.secrets["SUPABASE_DB_CONNECTION"]
    
    # Streamlit Cloud 환경에서 네트워크 및 SSL 연결 문제를 방지하기 위해 옵션을 안전하게 추가합니다.
    if "sslmode" not in raw_db_connection:
        if "?" in raw_db_connection:
            DB_CONNECTION = f"{raw_db_connection}&sslmode=require"
        else:
            DB_CONNECTION = f"{raw_db_connection}?sslmode=require"
    else:
        DB_CONNECTION = raw_db_connection

except KeyError as e:
    st.error(f"Secrets 설정이 누락되었습니다. 누락된 키: {e}")
    st.stop()

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# LlamaIndex 전역 설정 (Gemini 통합 사용)
@st.cache_resource
def init_llama_index():
    Settings.llm = GoogleGenAI(
        model="models/gemini-2.0-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.2
    )
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/text-embedding-004",
        api_key=GEMINI_API_KEY
    )

init_llama_index()

st.title("💼 조달청 RFP 분석 및 제안서 생성 AI Agent")
st.markdown("나라장터 제안요청서(RFP)를 분석하여 제안서 초안 작성 및 요구사항 충족률을 검증합니다.")

tab1, tab2, tab3 = st.tabs(["📥 RFP 업로드 및 설정", "✨ 제안서 생성 & 검토", "📜 히스토리"])

# 세션 상태 초기화
if "index" not in st.session_state:
    st.session_state.index = None
if "rfp_name" not in st.session_state:
    st.session_state.rfp_name = None

# =====================
# 탭 1: RFP 업로드 및 설정
# =====================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 제안사(우리 회사) 정보 입력")
        company_name = st.text_input("회사명", value="미래 혁신 테크", key="comp_name")
        company_spec = st.text_area(
            "회사 주요 역량 및 기술 스택",
            value="클라우드 네이티브 아키텍처 구축 전문, MSA 설계 노하우 보유, AI/ML 기반 데이터 플랫폼 개발 역량.",
            height=150
        )
        if st.button("💾 회사 정보 저장"):
            st.success("회사 역량 정보가 에이전트에 반영되었습니다.")

    with col2:
        st.subheader("📄 공공 RFP(제안요청서) 파일 업로드")
        rfp_file = st.file_uploader("나라장터 RFP PDF 파일을 업로드하세요", type=["pdf"])
        
        if rfp_file is not None:
            st.info(f"업로드 완료: {rfp_file.name}")
            
            if st.button("🔍 RFP 지식화 및 인덱싱 시작"):
                with st.spinner("LlamaIndex 가동 중... PDF 분석 및 Supabase pgvector 저장 중입니다."):
                    try:
                        # 임시 폴더에 파일 저장
                        with tempfile.TemporaryDirectory() as tmpdir:
                            filepath