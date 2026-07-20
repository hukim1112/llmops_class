from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from app.prompts import BASIC_SYSTEM_PROMPT

def get_agent_executor():
    # LLM (No tools)
    llm = init_chat_model(model="gpt-4o", model_provider="openai")
    
    # Memory
    memory = MemorySaver()
    
    # 오늘 날짜 기반으로 System Prompt 생성
    today_date = date.today().strftime("%Y-%m-%d")
    system_prompt = BASIC_SYSTEM_PROMPT.format(today_date=today_date)
    
    # Create Basic Agent (도구가 없는 순수 LLM 챗봇)
    basic_agent = create_agent(
        model=llm, # No tools bound
        tools=[], 
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    return basic_agent

agent_executor = get_agent_executor()
