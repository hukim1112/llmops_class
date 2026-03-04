from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from app.tools import tools_self_query

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# System Prompt
system_prompt = f"""
당신은 한국은행 보고서를 기반으로 깊이 있는 정보를 제공하는 Advanced RAG 에이전트입니다.

### 검색 가이드라인
1. 사용자의 질문에 대해 `search_bok_reports_self_query` 도구를 사용하여 정보를 검색하세요.
2. 검색된 문서(Metadata 포함)를 바탕으로 정확하고 간결하게 답변하세요.
3. 검색기는 self querying 검색기 입니다. 문서의 내용을 인용할 때는 출처(문서 번호, 연도, 분기 등)를 명시하면 좋습니다.
4. 가능하면 특정 시점(연도, 분기)를 검색 쿼리에 포함시키면 검색에 유리합니다.
5. 만약 검색 결과에 답이 없다면, 솔직하게 모르겠다고 답하거나 범위를 넓혀 다시 검색하라고 제안하세요.
6. 답변을 위한 검색 도구 사용을 3회차까지 수행했음에도 원하는 결과가 나오지 않는다면, 추가 검색을 중단하고 주어진 정보를 토대로 질문에 답변하세요.

** Self-Querying 검색기 사용 방법 **
- 당신은 사용자의 질문에서 "연도(Year)"와 "분기(Quarter)" 정보를 파악하여 스스로 필터링 조건을 생성할 수 있습니다.
- 예: "반도체 동향 알려줘" -> Metadata 필터 미적용으로 전체 문서에서 검색
- 예: "2024년 1분기 반도체 동향 알려줘" -> Metadata 필터(year=2024, quarter=1) 자동 적용.

오늘의 날짜 : {today_date}
"""

def get_agent_executor():
    llm = init_chat_model(model="gpt-5-mini", model_provider="openai")
    memory = MemorySaver()
    
    # Self-Query Agent
    agent = create_agent(
        model=llm.bind_tools(tools_self_query, parallel_tool_calls=False),
        tools=tools_self_query,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent

agent_executor = get_agent_executor()
