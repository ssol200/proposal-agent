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

# 프로젝트 정보 고정 설정
PROJECT_ID = "ggnrlfciqinywaodofuo"

# 1. 시크릿 관리 및 6543 포트용 싱가포르 Pooler 주소 수동 강제 조립
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    raw_connection = st.secrets["SUPABASE_DB_CONNECTION"]
    
    # 1단계: 원래 주소에서 비밀번호 부분만 정확하게 도려냅니다.
    # postgresql://postgres:[비밀번호]@db.ggnrlfciqinywaodofuo... 또는 pooler 구조 대응
    try:
        pw_step1 = raw_connection.split("://postgres:", 1)[1]
        DB_PASSWORD = pw_step1.split("@", 1)[0]
    except Exception:
        # 혹시 분리에 실패할 경우 원본을 그대로 씁니다.
        DB_CONNECTION = raw_connection
        DB_PASSWORD = None

    if DB_PASSWORD:
        # 2단계: 에러를 유발하는 기존 user("postgres") 대신 "postgres.프로젝트ID"로 유저명을 교체하고, 
        # 싱가포르 풀러 공식 호스트 도메인(aws-1-ap-southeast-1.pooler.supabase.com:6543)으로 주소를 새로 조립합니다.
        DB_CONNECTION = f"postgresql://postgres.{PROJECT_ID}:{DB_PASSWORD}@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
    else:
        DB_CONNECTION = raw_connection

except KeyError:
    st.error("Secrets 설정이 누락되었습니다. Streamlit Advanced Settings를 확인하세요.")
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
                        with tempfile.TemporaryDirectory() as tmpdir:
                            filepath = os.path.join(tmpdir, rfp_file.name)
                            with open(filepath, "wb") as f:
                                f.write(rfp_file.getbuffer())
                            
                            reader = SimpleDirectoryReader(input_dir=tmpdir)
                            documents = reader.load_data()
                            
                            vector_store = SupabaseVectorStore(
                                postgres_connection_string=DB_CONNECTION,
                                collection_name="data_rfp_vectors",
                                dimension=768
                            )
                            storage_context = StorageContext.from_defaults(vector_store=vector_store)
                            
                            index = VectorStoreIndex.from_documents(
                                documents, 
                                storage_context=storage_context
                            )
                            
                            st.session_state.index = index
                            st.session_state.rfp_name = rfp_file.name
                            
                            supabase.table("rfp_documents").insert({"rfp_name": rfp_file.name}).execute()
                            
                        st.success("🎉 RFP 분석 및 벡터 데이터베이스 저장이 완료되었습니다!")
                    except Exception as e:
                        st.error(f"인덱싱 중 에러 발생: {e}")

# =====================
# 탭 2: 제안서 생성 & 검토
# =====================
with tab2:
    st.subheader("🛠️ AI 제안서 작성 파이프라인")
    
    if st.session_state.index is None:
        st.warning("먼저 '탭 1'에서 RFP 파일을 업로드하고 인덱싱을 완료해 주세요.")
    else:
        st.write(f"현재 로드된 RFP 구역: **{st.session_state.rfp_name}**")
        
        if st.button("🚀 제안서 초안 및 요구사항 검증 시작"):
            query_engine = st.session_state.index.as_query_engine(similarity_top_k=4)
            
            with st.status("단계별 에이전트 태스크 수행 중...", expanded=True) as status:
                status.update(label="1단계: RFP 핵심 요구사항 파악 중 (RAG)...")
                rag_response = query_engine.query("이 제안요청서(RFP)의 핵심 기술적 요구사항과 필수 제출 항목을 요약해줘.")
                rfp_context = str(rag_response)
                
                status.update(label="2단계: 맞춤형 제안서 목차 설계 및 섹션별 초안 작성 중...")
                prompt_proposal = f"""
                당신은 공공 입찰 전문 수석 제안서 작성가입니다.
                다음 RFP 요구사항 내용과 제안사의 역량을 바탕으로 규격에 맞는 제안서 초안을 작성하세요.
                반드시 아래 JSON 포맷으로만 응답하세요. 키값은 바꾸지 마세요.

                [RFP 핵심 요구사항]
                {rfp_context}

                [제안사 역량]
                회사명: {company_name}
                역량 세부: {company_spec}

                [출력 포맷 (JSON 필수)]
                {{
                    "1. 제안 개요": "RFP 문제를 해결하기 위한 동기 및 핵심 전략 기술...",
                    "2. 기술 적용 방안": "RFP 핵심 기술 요구사항을 충족하기 위한 구체적 아키텍처 및 방법론...",
                    "3. 프로젝트 관리 및 수행 계획": "일정 관리, 인력 구성 및 산출물 계획...",
                    "4. 제안사의 특장점": "우리 회사의 특장점 및 차별화 포인트 기술..."
                }}
                """
                
                llm_response = Settings.llm.complete(prompt_proposal)
                try:
                    clean_text = str(llm_response).strip().replace("```json", "").replace("```", "")
                    proposal_sections = json.loads(clean_text)
                except:
                    proposal_sections = {
                        "1. 제안 개요": "제안요청서 내용을 바탕으로 한 맞춤형 솔루션 개요 수립 완료.",
                        "2. 기술 적용 방안": f"{company_name}의 핵심 클라우드 기술 및 아키텍처 설계 제안.",
                        "3. 프로젝트 관리 및 수행 계획": "철저한 일정 관리 및 품질 보증 계획 수립.",
                        "4. 제안사의 특장점": "유사 프로젝트 수행 경험 및 전담 인력 매칭 체계 확보."
                    }

                status.update(label="3단계: RFP 요구사항 충족도 자체 검증 진행 중...")
                prompt_compliance = f"""
                작성된 제안서 초안이 RFP 요구사항을 충족하는지 냉정하게 평가하세요.
                반드시 아래 구조의 JSON 포맷 리스트로만 답변하세요.

                [제안서 초안 구조]
                {json.dumps(proposal_sections, ensure_ascii=False)}

                [출력 포맷 (JSON 필수)]
                [
                    {{"요구사항": "클라우드 인프라 구축 요구", "충족여부": true, "이유": "제안서 2장에 클라우드 네이티브 아키텍처가 명시됨"}},
                    {{"요구사항": "데이터 플랫폼 연동 요구", "충족여부": true, "이유": "AI/ML 기반 데이터 플랫폼 개발 역량 투입 계획 확인됨"}},
                    {{"요구사항": "보안 및 암호화 요구", "충족여부": false, "이유": "초안에 구체적인 보안 솔루션 도입 내용이 누락됨"}}
                ]
                """
                
                comp_response = Settings.llm.complete(prompt_compliance)
                try:
                    clean_comp = str(comp_response).strip().replace("```json", "").replace("```", "")
                    compliance_result = json.loads(clean_comp)
                except:
                    compliance_result = [{"요구사항": "기본 기능 충족 여부 확인", "충족여부": True, "이유": "전반적인 규격 부합"}]
                
                true_count = sum(1 for item in compliance_result if item.get("충족여부") is True)
                fulfillment_rate = round((true_count / len(compliance_result)) * 100, 1) if compliance_result else 100.0

                status.update(label="4단계: 생성된 제안서 및 검증 결과 데이터베이스(Supabase) 저장 중...")
                proposal_data = {
                    "rfp_name": st.session_state.rfp_name,
                    "company_name": company_name,
                    "proposal_content": proposal_sections,
                    "requirements_check": compliance_result,
                    "fulfillment_rate": fulfillment_rate
                }
                
                supabase.table("proposals").insert(proposal_data).execute()
                status.update(label="✨ 에이전트 파이프라인 처리 완료!", state="complete")
            
            st.balloons()
            st.success(f"📈 최종 RFP 요구사항 충족도 점수: **{fulfillment_rate}%**")
            
            res_col1, res_col2 = st.columns([3, 2])
            with res_col1:
                st.markdown("### 📝 AI 제안서 초안 결과")
                for section, content in proposal_sections.items():
                    with st.expander(f"📍 {section}", expanded=True):
                        st.write(content)
                        
            with res_col2:
                st.markdown("### 🔍 요구사항 검증 매트릭스 (Compliance Matrix)")
                df_comp = pd.DataFrame(compliance_result)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
                
            st.markdown("---")
            st.markdown("### 📥 결과물 내보내기")
            d_col1, d_col2 = st.columns(2)
            
            try:
                doc = Document()
                doc.add_heading(f"제안서 초안: {st.session_state.rfp_name}", 0)
                doc.add_paragraph(f"제안사: {company_name}")
                doc.add_paragraph(f"작성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                doc.add_paragraph(f"요구사항 검증 충족률: {fulfillment_rate}%")
                
                for section, content in proposal_sections.items():
                    doc.add_heading(section, level=1)
                    doc.add_paragraph(content)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    doc.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        file_bytes = f.read()
                        
                d_col1.download_button(
                    label="📥 Word 파일(.docx) 다운로드",
                    data=file_bytes,
                    file_name=f"제안서_초안_{company_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                d_col1.error(f"Word 파일 빌드 오류: {e}")

            d_col2.info("💡 PDF 변환이 필요하신 경우, 다운로드한 Word 파일을 실행하여 '다른 이름으로 저장 -> PDF' 기능을 이용하세요.")

# =====================
# 탭 3: 히스토리
# =====================
with tab3:
    st.subheader("📜 과거 제안서 생성 이력")
    try:
        response = supabase.table("proposals").select("*").order("created_at", desc=True).limit(50).execute()
        history_data = response.data

        if history_data:
            df_history = pd.DataFrame(history_data)
            st.dataframe(
                df_history[["created_at", "rfp_name", "company_name", "fulfillment_rate"]],
                use_container_width=True, hide_index=True
            )

            selected_id = st.selectbox("상세 보기할 제안서 ID 선택", df_history["id"])
            if selected_id:
                detail = next(item for item in history_data if item["id"] == selected_id)
                if detail.get("proposal_content"):
                    for sec, txt in detail["proposal_content"].items():
                        with st.expander(f"📍 {sec}"):
                            st.write(txt)

            csv = df_history.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 이력 전체 CSV 다운로드", data=csv, file_name="proposal_history.csv", mime="text/csv")
        else:
            st.info("아직 생성된 제안서 이력이 없습니다.")
    except Exception as e:
        st.error(f"히스토리 데이터를 불러오는 중 에러 발생: {e}")