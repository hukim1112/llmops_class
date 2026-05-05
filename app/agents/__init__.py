"""
에이전트 레지스트리 (Agent Registry).

=============================================================================
🏭 에이전트 팩토리 패턴
=============================================================================
이 모듈은 에이전트를 선언적으로 등록하고, 서버/UI/Client가 이 레지스트리를
읽어 자동으로 라우터·메뉴·CLI 목록을 생성하는 **에이전트 팩토리** 역할을 합니다.

새 에이전트를 추가하려면:
  1. app/agents/ 에 새 파일(예: my_agent.py)을 생성하고 agent_executor를 정의
  2. 아래 AGENT_REGISTRY에 딕셔너리 항목 1개만 추가
  → 서버, UI, Client 모두 자동으로 반영됩니다.
"""

AGENT_REGISTRY = [
    {
        "name": "basic",
        "module": "app.agents.basic",
        "prefix": "/basic",
        "tags": ["Basic Chat"],
        "description": "도구 없이 LLM과 대화하는 기본 챗봇",
    },
    {
        "name": "basic-rag",
        "module": "app.agents.rag_basic",
        "prefix": "/basic-rag",
        "tags": ["RAG Basic"],
        "description": "한국은행 보고서 기반 기본 RAG",
    },
    {
        "name": "self-query-rag",
        "module": "app.agents.rag_self_query",
        "prefix": "/self-query-rag",
        "tags": ["Self-Query"],
        "description": "메타데이터 자동 필터링 RAG",
    },
    {
        "name": "multimodal-rag",
        "module": "app.agents.rag_multimodal",
        "prefix": "/multimodal-rag",
        "tags": ["Multimodal-RAG"],
        "description": "이미지 검색/분석 멀티모달 RAG",
    },
]
