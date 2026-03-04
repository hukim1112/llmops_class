import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# 1. 환경 변수 로드 (.env 파일 읽기)
load_dotenv()

# 2. 프로젝트 루트 경로 동적 설정
# 이 스크립트가 위치한 폴더의 절대 경로를 project_root로 지정합니다.
project_root = Path(__file__).resolve().parent

# 시스템 경로에 프로젝트 루트 추가 (app 모듈 import 에러 방지용)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 관련 라이브러리 및 커스텀 모듈 가져오기
from langchain_core.tracers.context import tracing_v2_enabled
from app.agents.rag_self_query import agent_executor as target_agent
from app.utils.evaluator import run_ragas_evaluation


import pandas as pd

async def run_test():
    """RAG 파이프라인 성능을 비동기로 평가하는 메인 함수입니다."""
    
    dataset_csv_path = os.path.join(project_root, "data", "evaluation", "golden_dataset.csv")
    output_result_path = os.path.join(project_root, "data", "evaluation", "result_self_query_rag.csv")
    
    print(f"🚀 [평가 시작] Golden Dataset을 불러옵니다: {dataset_csv_path}")
    
    # LangSmith 트레이싱(추적) 기능 켜기: 실행 내역이 LangSmith 사이트 프로젝트 이름에 기록됨
    with tracing_v2_enabled(project_name="test_self_query_rag"):
        # RAGAS 기반 비동기 평가 모듈 실행
        results = await run_ragas_evaluation(
            agent_executor=target_agent,
            dataset_path=dataset_csv_path,
            output_file=output_result_path
        )
        
    print("\n==================================")
    print("📈 [결과 확인] 최종 평가 점수 산출 완료")
    print("==================================")
    print(results)
    print(f"\n✅ 평가 결과표가 CSV 형식으로 저장되었습니다: {output_result_path}")


if __name__ == "__main__":
    # 비동기 함수 실행을 위한 이벤트 루프 시작
    asyncio.run(run_test())
