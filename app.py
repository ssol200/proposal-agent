import streamlit as st
import json
import tempfile
import io
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# 기본 레이아웃 구성
st.set_page_config(page_title="RFP 제안서 분석 & 생성 에이전트", layout="wide")

# 1. API 키 연동
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Secrets에 GEMINI_API_KEY 설정이 누락되었습니다. Streamlit 설정을 확인하세요.")
    st.stop()

# 구글 공식 최신 GenAI 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)

st.title("💼 조달청 RFP 분석 및 제안서 생성 AI Agent")
st.markdown("데이터베이스 연결 없이 무겁지 않고 빠르게 RFP를 분석하고 제안서 초안을 만듭니다.")

tab1, tab2 = st.tabs(["📥 RFP 업로드 및 설정", "✨ 제안서 생성 & 검토"])

# 세션 상태 초기화 (메모리 저장소)
if "rfp_text" not in st.session_state:
    st.session_state.rfp_text = None
if "rfp_name" not in st.session_state:
    st.session_state.rfp_name = None
if "proposal_result" not in st.session_state:
    st.session_state.proposal_result = None

# ==========================================
# 탭 1: 문서 업로드 및 텍스트 추출
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏢 제안사 정보 입력")
        company_name = st.text_input("회사명", value="미래 혁신 테크", key="comp_name")
        company_spec = st.text_area(
            "회사 주요 역량 및 기술 스택", 
            value="클라우드 네이티브 아키텍처 구축 전문, MSA 설계 노하우 보유, AI/ML 기반 데이터 플랫폼 개발 역량.", 
            height=150
        )
            
    with col2:
        st.subheader("📄 공공 RFP 파일 업로드")
        rfp_file = st.file_uploader("RFP PDF 파일을 업로드하세요", type=["pdf"])
        
        if rfp_file is not None:
            st.info(f"선택된 파일: {rfp_file.name}")
            if st.button("🔍 RFP 텍스트 분석 시작"):
                with st.spinner("PDF 파일에서 핵심 텍스트를 추출하는 중..."):
                    try:
                        pdf_reader = PdfReader(rfp_file)
                        extracted_text = ""
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + "\n"
                        
                        if len(extracted_text.strip()) == 0:
                            st.error("PDF에서 텍스트를 읽을 수 없습니다. 스캔된 이미지 형태의 PDF인지 확인해 주세요.")
                        else:
                            st.session_state.rfp_text = extracted_text
                            st.session_state.rfp_name = rfp_file.name
                            st.success("🎉 RFP 파일의 문맥 분석이 완료되었습니다! '제안서 생성 & 검토' 탭으로 이동하세요.")
                    except Exception as e:
                        st.error(f"파일 읽기 에러 발생: {e}")

# ==========================================
# 탭 2: 제안서 빌드 파트
# ==========================================
with tab2:
    st.subheader("🛠️ AI 제안서 작성 파이프라인")
    if st.session_state.rfp_text is None:
        st.warning("탭 1에서 RFP 파일을 먼저 업로드하고 텍스트 분석을 완료해 주세요.")
    else:
        st.write(f"현재 대상 문서: **{st.session_state.rfp_name}**")
        
        if st.button("🚀 제안서 초안 및 요구사항 검증 시작"):
            with st.status("Gemini 2.0 Flash가 RFP를 분석하여 제안서를 작성 중입니다...", expanded=True) as status:
                
                status.update(label="1단계: 용량 초과 방지 필터링 및 AI 초안 작성 중...")
                
                # ⚠️ 핵심 안전 장치: 대용량 PDF 압박으로 인한 에러를 방지하기 위해 글자 수 제한을 엄격하게 조정합니다.
                safe_rfp_context = st.session_state.rfp_text[:4000]
                
                prompt = f"""
                당신은 공공기관 조달청 제안서 작성 전문 AI 에이전트입니다.
                다음 제공된 [RFP 내용]을 분석하고, [제안사 역량]을 매칭하여 최적의 제안서를 작성해 주세요.
                
                [RFP 내용]
                {safe_rfp_context}
                
                [제안사 역량]
                {company_spec}
                
                출력은 다른 설명이 전혀 없는 순수한 JSON 텍스트여야 하며, 반드시 다음 구조를 완벽히 준수해야 합니다.
                {{
                    "proposal_content": {{
                        "1. 제안 개요": "RFP 요구사항에 대응하는 제안 목적과 가치 상세",
                        "2. 기술 적용 방안": "제안사의 기술을 활용한 시스템 구현 방안",
                        "3. 프로젝트 수행 계획": "일정 관리 및 품질 보증 방안"
                    }},
                    "compliance_check": [
                        {{"요구사항": "핵심 기술 요구사항 만족 여부", "충족여부": true, "이유": "역량에 기반하여 요구 규격을 충족함"}}
                    ],
                    "fulfillment_rate": 95.0
                }}
                """
                
                try:
                    response = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.3
                        )
                    )
                    
                    # 텍스트를 깨끗하게 청소한 후 파싱
                    response_text = response.text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text.replace("```json", "", 1)
                    if response_text.endswith("```"):
                        response_text = response_text.rsplit("```", 1)[0]
                        
                    result_json = json.loads(response_text.strip())
                    st.session_state.proposal_result = result_json
                    status.update(label="✨ 제안서 및 검증 매트릭스 생성 성공!", state="complete")
                    
                except Exception as ai_err:
                    status.update(label="❌ 생성 중 에러 발생", state="error")
                    st.error(f"상세 분석 에러 내용: {ai_err}")
            
            # 결과 화면 출력
            if st.session_state.proposal_result:
                res_data = st.session_state.proposal_result
                st.balloons()
                
                st.success(f"📈 RFP 요구사항 충족도 점수: **{res_data.get('fulfillment_rate', 100)}%**")
                
                res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.markdown("### 📝 AI 제안서 초안")
                    proposal_sections = res_data.get("proposal_content", {})
                    for section, content in proposal_sections.items():
                        with st.expander(f"📍 {section}", expanded=True):
                            st.write(content)
                with res_col2:
                    st.markdown("### 🔍 요구사항 검증 매트릭스")
                    compliance_list = res_data.get("compliance_check", [])
                    import pandas as pd
                    st.dataframe(pd.DataFrame(compliance_list), use_container_width=True, hide_index=True)
                    
                st.markdown("---")
                
                # 워드 파일(.docx) 다운로드 기능
                try:
                    doc = Document()
                    doc.add_heading(f"제안서 초안: {st.session_state.rfp_name}", 0)
                    for section, content in proposal_sections.items():
                        doc.add_heading(section, level=1)
                        doc.add_paragraph(content)
                    
                    bio = io.BytesIO()
                    doc.save(bio)
                    bio.seek(0)
                    
                    st.download_button(
                        label="📥 Word 파일(.docx) 다운로드", 
                        data=bio.getvalue(), 
                        file_name=f"{company_name}_제안서_초안.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except Exception as doc_err:
                    st.error(f"다운로드 파일 변환 실패: {doc_err}")