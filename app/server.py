# 리눅스 환경에서 시스템 SQLite 버전이 낮을 경우 pysqlite3를 대신 사용하도록 강제함
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sys
import os
from dotenv import load_dotenv

# 1. Setup Project Root Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 2. 환경 변수 로드 (.env 파일 명시적 지정)
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path)
    
    # LangSmith Project Setting (Server Specific)
    # .env에는 API Key만 있고, 프로젝트명은 여기서 분리합니다.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "llmops-agent-server"
    print(f"📈 LangSmith Tracing Enabled. Project: {os.environ['LANGSMITH_PROJECT']}")
else:
    print("Warning: .env file not found.")

import logging
import json
import traceback
from typing import AsyncGenerator, Optional, Dict, Any

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- Agent Executors Import ---
# 주의: 환경 변수(API Key)가 로드된 후에 에이전트 모듈을 임포트해야 합니다.
from app.agents.basic import agent_executor as basic_agent
from app.agents.rag_basic import agent_executor as rag_basic_agent
from app.agents.rag_self_query import agent_executor as rag_self_query_agent
from app.agents.rag_multimodal import agent_executor as rag_multimodal_agent

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMOps_Server")

# --- Schemas ---
class UserInput(BaseModel):
    message: str
    thread_id: Optional[str] = None

class StreamInput(UserInput):
    stream_tokens: bool = Field(default=True)

class ChatMessage(BaseModel):
    type: str
    content: str


# --- Router Factory ---
def create_agent_router(agent_executor, prefix: str, tags: list = None) -> APIRouter:
    """
    주어진 에이전트 실행기(Executor)를 위한 FastAPI 라우터를 생성하는 팩토리 함수입니다.
    /invoke 및 /stream 엔드포인트를 자동으로 등록합니다.
    """
    router = APIRouter(prefix=prefix, tags=tags or [prefix])

    async def _stream_generator(input_data: StreamInput) -> AsyncGenerator[str, None]:
        try:
            config = {"configurable": {"thread_id": input_data.thread_id}} if input_data.thread_id else {}
            
            # LangGraph astream_events (v2)
            async for event in agent_executor.astream_events(
                {"messages": [("user", input_data.message)]}, 
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                # Tool Start
                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name'], 'input': event['data'].get('input')})}\n\n"
                
                # Token Streaming (Chat Model)
                elif kind == "on_chat_model_stream":
                    # 내부 로직(예: Self-Query 구성 등)에서 발생하는 중간 단계의 토큰은 제외합니다.
                    tags = event.get("tags", [])
                    if "exclude_from_stream" in tags:
                        continue

                    chunk = event["data"]["chunk"]
                    if chunk and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        except Exception as e:
            logger.error(f"Stream error in {prefix}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "event: end\ndata: \n\n"

    @router.post("/invoke", response_model=ChatMessage)
    async def invoke(input_data: UserInput):
        try:
            config = {"configurable": {"thread_id": input_data.thread_id}} if input_data.thread_id else {}
            
            # invoke returns the final state
            result = await agent_executor.ainvoke(
                {"messages": [("user", input_data.message)]},
                config=config
            )
            # LangGraph: State['messages'][-1] is the AI response
            last_message = result["messages"][-1]
            return ChatMessage(type="ai", content=last_message.content)
        except Exception as e:
            logger.error(f"Invocation error in {prefix}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/stream")
    async def stream(input_data: StreamInput):
        return StreamingResponse(
            _stream_generator(input_data), 
            media_type="text/event-stream"
        )
        
    return router


# --- App Initialization ---
app = FastAPI(
    title="LLMOps Class Agent Server", 
    version="1.0",
    description="Unified Server for Multiple RAG Agents"
)

@app.get("/health")
def health():
    return {"status": "ok", "agents": ["basic", "basic-rag", "self-query-rag", "multimodal-rag"]}

# --- Register Routers ---
app.include_router(create_agent_router(basic_agent, "/basic", ["Basic Chat"]))
app.include_router(create_agent_router(rag_basic_agent, "/basic-rag", ["RAG Basic"]))
app.include_router(create_agent_router(rag_self_query_agent, "/self-query-rag", ["Self-Query"]))
app.include_router(create_agent_router(rag_multimodal_agent, "/multimodal-rag", ["Multimodal-RAG"]))


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Server Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server Host")
    args = parser.parse_args()
    
    print(f"🚀 Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
