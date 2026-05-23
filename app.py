import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from docx import Document
from fpdf import FPDF
from supabase import create_client, Client

# LlamaIndex & LangChain 관련
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputParser

# 1. 시크릿 및 설정 관리
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]
except KeyError:
    st.error("Secrets 설정이 누락되었습니다. .streamlit/secrets.toml 확인 필요.")
    st.stop()

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Gemini LLM 설정 (Agent용)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)

# LlamaIndex 설정 (RAG용)
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GEMINI_API_KEY)
llama_llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)

# 페이지 설정
st.set_page_config(page_title="RFP 제안서 자동생성 AI Agent", layout="wide")

# 2. 세션 상태 초기화
if "company_info" not in st.session_state:
    st.session_state.company_info = {}
if "proposal_sections" not in st.session_state:
    st.session_state.proposal_sections = {}
if "requirements_check" not in st.session_state:
    st.session_state.requirements_check = []
if "fulfillment_rate" not in st.session_state:
    st.session_state.fulfillment_rate = 0.0

# --- 공통 함수 영역 ---

def save_to_supabase(rfp_name, company_name, content, check, rate):
    """결과물을 Supabase에 저장함"""
    data = {
        "rfp_name": rfp_name,
        "company_name": company_name,
        "proposal_content": content,
        "requirements_check": check,
        "fulfillment_rate": rate
    }
    supabase.table("proposals").insert(data).execute()

# --- 4단계 Agent 파이프라인 함수 ---

def run_proposal_agent(query_engine, rfp_text_summary):
    # 1단계: RFP 분석 (LangChain)
    st.info("🔍 1단계: RFP 핵심 요구사항 분석 중...")
    analysis_prompt = PromptTemplate.from_template(
        "RFP 내용을 바탕으로 핵심 요구사항, 평가기준, 필수 포함 섹션을 추출해줘. JSON 형식으로 응답해.\n{context}"
    )
    analysis_chain = analysis_prompt | llm | JsonOutputParser()
    analysis_result = analysis_chain.invoke({"context": rfp_text_summary})

    # 2단계: 목차 설계
    st.info("📋 2단계: 제안서 목차 구성 중...")
    outline_prompt = PromptTemplate.from_template(
        "분석결과 {analysis}를 바탕으로 제안서 목차를 JSON 리스트(sections)로 생성해줘."
    )
    outline_chain = outline_prompt | llm | JsonOutputParser()
    outline_result = outline_chain.invoke({"analysis": json.dumps(analysis_result)})

    # 3단계 & 4단계 (루프)
    proposal_drafts = {}
    requirement_results = []
    loop_count = 0
    max_loops = 3
    final_rate = 0.0

    while loop_count < max_loops:
        loop_count += 1
        st.write(f"✍️ 3~4단계 실행 중 (시도 {loop_count}/{max_loops})...")
        
        # 3단계: 섹션별 초안 생성
        for section in outline_result['sections']:
            # RAG를 이용해 관련 문서 검색
            response = query_engine.query(f"{section}에 대한 구체적인 요구사항과 지침을 알려줘.")
            
            draft_prompt = PromptTemplate.from_template(
                """RFP 내용: {rfp_context}
                우리 회사 정보: {company}
                섹션명: {section_name}
                
                위 내용을 바탕으로 제안서 초안을 작성해.
                - 반드시 명사형 종결체('~함', '~임', '~예정')를 사용함.
                - 정보가 부족한 경우 [추가 정보 필요: 이유]라고 표기함.
                - 전문적이고 신뢰감 있는 톤을 유지함.
                초안 내용만 출력해."""
            )
            draft_chain = draft_prompt | llm
            draft = draft_chain.invoke({
                "rfp_context": response.response,
                "company": st.session_state.company_info,
                "section_name": section
            })
            proposal_drafts[section] = draft.content

        # 4단계: 요구사항 충족 검토
        st.info(f"⚖️ 요구사항 충족률 검토 중...")
        review_prompt = PromptTemplate.from_template(
            "RFP 요구사항 {reqs} 대비 작성된 제안서 {drafts}의 충족 여부를 리스트로 판별하고 전체 충족률(%)을 JSON으로 반환해. "
            "JSON 형식: {{'check': [{{'requirement': '...', 'met': true/false}}], 'fulfillment_rate': 85.0}}"
        )
        review_chain = review_prompt | llm | JsonOutputParser()
        review_result = review_chain.invoke({
            "reqs": json.dumps(analysis_result),
            "drafts": json.dumps(proposal_drafts)
        })

        requirement_results = review_result['check']
        final_rate = review_result['fulfillment_rate']

        if final_rate >= 80.0:
            break
        else:
            st.warning(f"충족률({final_rate}%)이 낮아 미비한 섹션을 다시 보완합니다.")

    return proposal_drafts, requirement_results, final_rate

# --- UI 화면 구성 ---

st.title("🚀 RFP 제안서 자동 생성 AI Agent")

tab1, tab2, tab3 = st.tabs(["📂 RFP 업로드", "✍️ 제안서 생성", "💾 히스토리"])

# [탭 1: RFP 업로드]
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏢 업체 정보 입력")
        with st.form("company_form"):
            c_name = st.text_input("회사명", value="주식회사 에이아이테크")
            c_ceo = st.text_input("대표자명")
            c_perf = st.text_area("주요 실적", placeholder="최근 3년간 공공기관 프로젝트 5건 수행 등")
            c_core = st.text_area("핵심 역량", placeholder="LLM 파인튜닝 기술 보유, 대용량 트래픽 처리 경험")
            submitted = st.form_submit_button("정보 저장")
            if submitted:
                st.session_state.company_info = {"회사명": c_name, "대표자": c_ceo, "실적": c_perf, "역량": c_core}
                st.success("업체 정보가 저장되었습니다.")

    with col2:
        st.subheader("📄 RFP PDF 업로드")
        uploaded_file = st.file_uploader("제안요청서(PDF)를 업로드하세요.", type="pdf")
        
        if uploaded_file and st.session_state.company_info:
            if st.button("RFP 분석 및 인덱싱 시작"):
                with st.spinner("LlamaIndex가 문서를 분석 중입니다..."):
                    # 임시 저장
                    with open("temp_rfp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # LlamaIndex 로드 및 Supabase 벡터 저장
                    documents = SimpleDirectoryReader(input_files=["temp_rfp.pdf"]).load_data()
                    vector_store = SupabaseVectorStore(
                        postgres_connection_string=DB_CONNECTION,
                        collection_name="data_rfp_vectors"
                    )
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(
                        documents, storage_context=storage_context, embed_model=embed_model
                    )
                    
                    st.session_state.index = index
                    st.session_state.rfp_name = uploaded_file.name
                    st.success("RFP 인덱싱 완료! 이제 제안서 생성 탭으로 이동하세요.")

# [탭 2: 제안서 생성]
with tab2:
    if "index" not in st.session_state:
        st.warning("먼저 탭 1에서 RFP를 업로드하고 인덱싱해주세요.")
    else:
        if st.button("✨ 제안서 초안 자동 생성 시작"):
            query_engine = st.session_state.index.as_query_engine(llm=llama_llm)
            # 요약 정보 추출 (1단계용)
            rfp_summary = query_engine.query("이 RFP의 주요 목적과 사업 범위를 요약해줘.").response
            
            content, check, rate = run_proposal_agent(query_engine, rfp_summary)
            
            st.session_state.proposal_sections = content
            st.session_state.requirements_check = check
            st.session_state.fulfillment_rate = rate
            
            # DB 저장
            save_to_supabase(st.session_state.rfp_name, st.session_state.company_info['회사명'], content, check, rate)

        # 결과 표시
        if st.session_state.proposal_sections:
            st.divider()
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.subheader("📝 생성된 제안서 초안")
                for section, text in st.session_state.proposal_sections.items():
                    with st.expander(f"📍 {section}", expanded=True):
                        edited_text = st.text_area(f"{section} 내용 편집", value=text, height=200, key=f"edit_{section}")
                        st.session_state.proposal_sections[section] = edited_text
            
            with col_res2:
                st.subheader("📊 검토 결과")
                st.metric("요구사항 충족률", f"{st.session_state.fulfillment_rate}%")
                st.table(pd.DataFrame(st.session_state.requirements_check))

            # 다운로드 기능
            st.subheader("📥 결과물 다운로드")
            d_col1, d_col2 = st.columns(2)
            
            # Word 생성
            doc = Document()
            doc.add_heading(f"제안서: {st.session_state.rfp_name}", 0)
            for s, t in st.session_state.proposal_sections.items():
                doc.add_heading(s, level=1)
                doc.add_paragraph(t)
            doc_path = "proposal_draft.docx"
            doc.save(doc_path)
            
            with open(doc_path, "rb") as f:
                d_col1.download_button("Word(.docx) 다운로드", f, file_name=f"{st.session_state.rfp_name}_제안서.docx")

            # PDF 생성 (간단 구현용)
            pdf = FPDF()
            pdf.add_page()
            # 참고: fpdf2에서 한글 폰트를 사용하려면 .ttf 파일이 필요합니다.
            # pdf.add_font('Nanum', '', 'NanumGothic.ttf', unicode=True)
            # pdf.set_font('Nanum', size=12)
            pdf.set_font("Arial", size=12) 
            pdf.cell(200, 10, txt="Proposal Draft (Korean Font Required for PDF)", ln=True, align='C')
            pdf_path = "proposal_draft.pdf"
            pdf.output(pdf_path)
            
            with open(pdf_path, "rb") as f:
                d_col2.download_button("PDF 다운로드 (영문)", f, file_name=f"{st.session_state.rfp_name}_제안서.pdf")

# [탭 3: 히스토리]
with tab3:
    st.subheader("📜 과거 제안서 생성 이력")
    response = supabase.table("proposals").select("*").order("created_at", desc=True).execute()
    history_data = response.data
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history[["created_at", "rfp_name", "company_name", "fulfillment_rate"]])
        
        selected_id = st.selectbox("상세 보기할 제안서 ID 선택", df_history["id"])
        if selected_id:
            detail = next(item for item in history_data if item["id"] == selected_id)
            st.json(detail["proposal_content"])
            
        csv = df_history.to_csv(index=False).encode('utf-8-sig')
        st.download_button("이력 데이터를 CSV로 다운로드", csv, "proposal_history.csv", "text/csv")
    else:
        st.info("아직 생성된 제안서 이력이 없습니다.")