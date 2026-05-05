"""
LLM 모델별 메시지 포맷 차이를 통일하는 유틸리티.

문제 배경:
  - OpenAI: content = "string" (단일 문자열)
  - Gemini:  content = [{"type": "text", "text": "string"}, ...] (리스트)

이 차이로 인해 SSE 스트리밍, invoke 응답 등에서 타입 에러가 발생할 수 있습니다.
normalize_content()를 통해 어떤 모델의 응답이든 단일 문자열로 정규화합니다.
"""


def sanitize_text(text: str) -> str:
    """서로게이트 문자를 제거하여 JSON 직렬화 오류를 방지합니다.

    PDF에서 추출된 한국어 텍스트에 깨진 유니코드(lone surrogate, U+D800~U+DFFF)가
    포함될 수 있습니다. 이 문자들은 UTF-8로 인코딩할 수 없어 json.dumps() 시
    UnicodeEncodeError를 발생시킵니다. 이 함수는 해당 문자를 '�'(U+FFFD)로 대체합니다.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


def normalize_content(content) -> str:
    """다양한 LLM의 content 형식을 단일 문자열로 정규화합니다.

    지원하는 형식:
      - str: 그대로 반환 (OpenAI 등)
      - list[dict]: type별로 파싱 (Gemini 등)
        - {"type": "text", "text": "..."} → 텍스트 추출
        - {"type": "image_url", "image_url": {"url": "..."}} → 마커 삽입
      - list[str]: 각 항목을 줄바꿈으로 결합
      - 기타: str()로 변환

    Args:
        content: LLM 응답의 content 필드 (str, list, 또는 기타)

    Returns:
        정규화된 단일 문자열
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # 이미지 URL은 텍스트로 변환하지 않고 표시용 마커 삽입
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        text_parts.append("[이미지 (Base64)]")
                    else:
                        text_parts.append(f"![image]({url})")
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    return str(content)
