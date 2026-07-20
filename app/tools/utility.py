import os
import base64
import json
import mimetypes
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch, TavilyExtract

tavily_search = TavilySearch(max_results=3, topic="general")
tavily_extract = TavilyExtract(extract_depth="basic", include_images=False)

MAX_TOOL_OUTPUT_CHARS = 6000
summarizer_llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

def _summarize_if_long(text: str, query: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """검색 결과가 길면 gpt-4o-mini로 핵심만 요약합니다."""
    if len(text) <= max_chars:
        return text
    print(f"  📋 검색 결과 {len(text)}자 → gpt-4o-mini 요약 중...")
    summary = summarizer_llm.invoke(
        f"다음은 '{query}'에 대한 검색 결과입니다. "
        f"핵심 정보만 추출하여 {query}와 관련한 핵심 자료를 위주로 {MAX_TOOL_OUTPUT_CHARS}이내에서 구조화된 요약을 작성하세요. "
        f"반드시 출처 URL을 보존하세요.\n\n{text[:50000]}"
    )
    result = summary.content
    print(f"  ✅ 요약 완료: {len(text)}자 → {len(result)}자")
    return result


def extract_text_content(content) -> str:
    """AIMessage.content에서 순수 텍스트만 추출합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else part
            for part in content if isinstance(part, (dict, str))
        )
    return str(content)

@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """웹에서 최신 정보를 검색합니다. 결과가 길면 자동으로 핵심만 요약됩니다.

    Args:
        query: 검색할 쿼리 문자열
    """
    result = tavily_search.invoke({"query": query})
    raw = result if isinstance(result, str) else str(result)
    return _summarize_if_long(raw, query)


@tool(parse_docstring=True)
def web_extract(url: str) -> str:
    """특정 URL에서 상세 정보를 추출합니다. 결과가 길면 자동으로 핵심만 요약됩니다.

    Args:
        url: 정보를 추출할 웹페이지 URL
    """
    result = tavily_extract.invoke({"urls": [url]})
    raw = result if isinstance(result, str) else str(result)
    return _summarize_if_long(raw, url)


@tool
def read_image_and_analyze(image_path: str, query_hint: str = "이 이미지의 내용을 상세히 설명해줘.") -> str:
    """
    로컬 이미지 파일을 읽고, Vision AI를 사용하여 이미지의 내용을 텍스트로 상세히 분석하여 반환합니다.
    이미지 경로와 함께, 무엇을 중점적으로 봐야 할지 힌트(query_hint)를 줄 수 있습니다.

    Args:
        image_path (str): 분석할 이미지의 파일 경로
        query_hint (str): 이미지에서 중점적으로 파악해야 할 내용 (예: "차트의 수치 변화를 설명해줘")
    """

    if not os.path.exists(image_path):
        return f"Error: 파일을 찾을 수 없습니다. 경로: {image_path}"

    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type: mime_type = "image/png"

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        vision_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": f"당신은 유능한 이미지 분석가입니다. 상위 에이전트의 요청에 맞춰 이미지를 분석하세요. 만약 정확한 분석에 실패하거나, 요청한 내용과 이미지의 차이가 있다면 그 이유를 설명하여 상위 에이전트가 질의를 조정하는데 도움을 주세요.: {query_hint}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                    }
                ]
            )
        ]

        result = vision_llm.invoke(messages)
        return f"[이미지 분석 결과 - {os.path.basename(image_path)}]\n{result.content}"

    except Exception as e:
        return f"Error: 이미지 분석 중 오류 발생. {str(e)}"