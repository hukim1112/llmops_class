from datetime import date
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from app.tools import tools_basic
from app.prompts import RAG_BASIC_SYSTEM_PROMPT

def get_agent_executor():
    # LLM & Memory
    llm = init_chat_model(model="gpt-5-mini", model_provider="openai")
    memory = MemorySaver()
    
    # 오늘 날짜 기반으로 System Prompt 생성
    today_date = date.today().strftime("%Y-%m-%d")
    system_prompt = RAG_BASIC_SYSTEM_PROMPT.format(today_date=today_date)
    
    # Create Agent (LangGraph 기반 Standard Agent)
    rag_agent = create_agent(
        model=llm.bind_tools(tools_basic, parallel_tool_calls=False),
        tools=tools_basic,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    return rag_agent

agent_executor = get_agent_executor()
