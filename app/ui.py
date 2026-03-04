import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import re
import uuid
from app.client import AgentClient

# --- Page Config ---
st.set_page_config(page_title="LLMOps AI Chat", layout="wide")

# --- Initialize Client ---
@st.cache_resource
def get_client():
    return AgentClient(base_url="http://localhost:8000")

client = get_client()

# --- Helpers ---
def render_message_content(content):
    """
    텍스트 내의 <Render_Image> 태그를 파싱하여
    텍스트와 이미지를 순서대로 렌더링합니다.
    """
    # 이미지 태그 패턴: <Render_Image>경로</Render_Image>
    pattern = re.compile(r"<Render_Image>(.*?)</Render_Image>")
    
    # 태그를 기준으로 텍스트를 분할 (split하면 텍스트와 경로가 번갈아 나옴)
    parts = pattern.split(content)
    
    for i, part in enumerate(parts):
        # 짝수 인덱스는 일반 텍스트, 홀수 인덱스는 이미지 경로(그룹 캡처)
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            # 이미지 경로
            image_path = part.strip()
            if os.path.exists(image_path):
                st.image(image_path, caption=os.path.basename(image_path))
            else:
                st.error(f"Image not found: {image_path}")

# --- Initialize Session State ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 LLMOps Chat")
    
    # Agent Selector
    agent_name = st.radio(
        "Select Agent",
        ["basic", "basic-rag", "self-query-rag", "multimodal-rag"],
        index=0
    )
    
    st.markdown("---")
    st.caption(f"Thread ID: {st.session_state.thread_id}")
    if st.button("New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
st.subheader(f"Chat with `{agent_name}`")

# 1. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 저장된 메시지는 렌더링 함수를 통해 처리
        render_message_content(msg["content"])

# 2. Chat Input
if prompt := st.chat_input("메시지를 입력하세요..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Agent Response (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Streamlit은 스트리밍 중에 이미지를 중간중간 띄우기 까다로우므로
        # 텍스트가 완성된 후에 파싱해서 렌더링하는 방식이 안전합니다.
        # 혹은 청크 단위로 텍스트만 먼저 보여주다가 완료되면 리렌더링합니다.
        
        # A. 텍스트 스트리밍 수신 (Token 단위)
        for chunk in client.stream(agent_name, prompt, st.session_state.thread_id):
            if "type" in chunk:
                if chunk["type"] == "token":
                    content = chunk.get("content", "")
                    full_response += content
                    # 스트리밍 중에는 텍스트만 보여줌 (Raw 태그 포함)
                    message_placeholder.markdown(full_response + "▌")
                elif chunk["type"] == "tool_start":
                    with st.status(f"🛠️ 도구 사용 중: {chunk['name']}", expanded=False) as status:
                        st.write(f"Input: {chunk.get('input')}")
                        status.update(state="complete")
                elif chunk["type"] == "error":
                    st.error(f"Error: {chunk.get('content')}")
        
        # B. 완료 후 최종 렌더링 (이미지 태그 처리)
        message_placeholder.empty() # 기존 스트리밍 텍스트 지움 (Clean up)
        render_message_content(full_response) # 파싱 및 이미지 렌더링 (Parsing & Rendering)
        
        # Add Assistant Message to History
        st.session_state.messages.append({"role": "assistant", "content": full_response})
