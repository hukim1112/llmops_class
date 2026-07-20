from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from app.tools import tools_multimodal
from app.prompts import RAG_MULTIMODAL_SYSTEM_PROMPT

def get_agent_executor():
    llm = init_chat_model(model="gpt-5-mini", model_provider="openai")
    memory = MemorySaver()
    
    # 오늘 날짜 기반으로 System Prompt 생성
    today_date = date.today().strftime("%Y-%m-%d")
    system_prompt = RAG_MULTIMODAL_SYSTEM_PROMPT.format(today_date=today_date)
    
    agent = create_agent(
        model=llm.bind_tools(tools_multimodal, parallel_tool_calls=False),
        tools=tools_multimodal,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent

agent_executor = get_agent_executor()
