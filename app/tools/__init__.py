"""
도구 팩토리 (Tool Factory).

=============================================================================
🏭 도구 팩토리 패턴
=============================================================================
이 모듈은 에이전트별로 사용할 도구 세트를 중앙에서 관리합니다.
새 도구를 추가하려면:
  1. app/tools/ 에 도구 함수를 정의 (기존 rag.py, utility.py 또는 새 파일)
  2. 여기서 import 후 적절한 tools_* 리스트에 추가
  → 해당 에이전트가 자동으로 새 도구를 사용합니다.

=============================================================================
🔌 확장 가능성: MCP(Model Context Protocol) 통합
=============================================================================
현재는 도구를 정적으로 import하여 리스트에 추가하는 방식이지만,
MCP(Model Context Protocol)를 통합하면 외부 도구 서버와의 동적 연동이 가능합니다.

1. MCP 서버 연동 예시:
   - MCP 서버에 등록된 도구들을 런타임에 자동 검색(discovery)하여
     tools_* 리스트에 동적으로 추가할 수 있습니다.
   - 예) MCP 서버에 "금융 데이터 분석", "차트 생성", "번역" 도구가 등록되어 있으면
         해당 도구들을 에이전트에 자동 바인딩

2. AI 기반 도구 자동 추천 (Agent-Tool Matching):
   - AGENT_REGISTRY의 각 에이전트 description을 AI(LLM/Embedding)가 분석하여,
     MCP에 등록된 도구 목록 중 해당 에이전트에 적합한 도구들을 자동으로 추천·연동
     할 수 있습니다.
   - 예) description="이미지 검색/분석 멀티모달 RAG" 에이전트에는
         MCP의 "이미지 생성", "OCR", "차트 분석" 도구를 자동 매칭
   - 예) description="한국은행 보고서 기반 기본 RAG" 에이전트에는
         MCP의 "경제 지표 API", "환율 조회" 도구를 자동 매칭
   - 이를 통해 새 에이전트를 등록할 때 description만 잘 작성하면
     도구 바인딩까지 자동화되는 "Zero-Config Tool Binding"이 가능합니다.

3. 구현 스케치 (멀티 MCP 서버 통합):
   langchain-mcp-adapters의 MultiServerMCPClient를 활용하면
   여러 MCP 서버를 동시에 연결하고, 모든 도구를 통합 관리할 수 있습니다.

   ```
   # pip install langchain-mcp-adapters
   # from langchain_mcp_adapters.client import MultiServerMCPClient
   # from app.agents import AGENT_REGISTRY
   #
   # # 멀티 MCP 서버 연결 (각 서버는 독립 프로세스로 운영)
   # mcp_client = MultiServerMCPClient({
   #     "FinanceTools": {                        # 금융 데이터 MCP 서버
   #         "transport": "streamable_http",
   #         "url": "http://localhost:6001/mcp/",
   #     },
   #     "ImageTools": {                          # 이미지 처리 MCP 서버
   #         "transport": "streamable_http",
   #         "url": "http://localhost:6002/mcp/",
   #     },
   #     "SearchTools": {                         # 웹 검색 MCP 서버
   #         "transport": "streamable_http",
   #         "url": "http://localhost:6003/mcp/",
   #     },
   # })
   #
   # # 모든 MCP 서버의 도구를 한 번에 가져오기
   # mcp_tools = await mcp_client.get_tools()
   #
   # # AI 기반 자동 매칭: 에이전트 description으로 적합한 도구 추천
   # for agent in AGENT_REGISTRY:
   #     matched = ai_recommend_tools(agent["description"], mcp_tools)
   #     tools_registry[agent["name"]].extend(matched)
   #
   # # 에이전트 생성 시 로컬 도구 + MCP 도구 병합
   # all_tools = local_tools + mcp_tools
   # agent = create_agent(model=llm.bind_tools(all_tools), tools=all_tools, ...)
   ```
"""

from app.tools.rag import (
    search_bok_reports_basic, 
    search_bok_reports_self_query, 
    search_bok_reports_multimodal
)
from app.tools.utility import (
    read_image_and_analyze, 
    web_search_custom_tool
)

# Export Tool Lists for Agents
tools_basic = [search_bok_reports_basic]
tools_self_query = [search_bok_reports_self_query]
tools_multimodal = [search_bok_reports_multimodal, read_image_and_analyze, web_search_custom_tool]
