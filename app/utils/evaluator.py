import json
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS Imports
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness

# JSON Parsing Helper (RAGAS 안정성 확보용)
class JSONCleanLLM(ChatOpenAI):
    def _clean(self, text):
        if "```json" in text: return text.replace("```json", "").replace("```", "").strip()
        if "```" in text: return text.replace("```", "").strip()
        return text

    async def agenerate(self, messages, stop=None, **kwargs):
        result = await super().agenerate(messages, stop=stop, **kwargs)
        for gens in result.generations:
            for gen in gens:
                gen.text = self._clean(gen.text)
        return result

async def run_ragas_evaluation(
    agent_executor, 
    dataset_path: str, 
    output_file: str = "ragas_results.csv",
    project_name: str = "RAG_Evaluation"
):
    """
    CSV 데이터셋을 로드하여 Agent를 실행하고 RAGAS로 평가합니다.
    
    Args:
        agent_executor: 평가할 LangGraph Agent Executor
        dataset_path: Golden Dataset CSV 파일 경로
        output_file: 평가 결과를 저장할 CSV 경로
        project_name: (Optional) 로깅용 프로젝트 이름
    """
    
    print(f"📊 평가 시작: {dataset_path} -> Agent")
    
    # 1. 데이터셋 로드
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return None
        
    questions = df["question"].tolist()
    # RAGAS 최신 버전 SingleTurnSample 호환성 수정: 
    # reference(ground_truth)는 List[str] 대신 str로 전달 (단일 정답인 경우)
    ground_truths = df["answer"].tolist() 
    # ground_truth_contexts = [[ctx] for ctx in df["ground_truth_context"].tolist()] # Optional
    
    answers = []
    contexts_list = []
    
    # 2. Agent 실행 및 데이터 수집
    print(f"🚀 총 {len(questions)}개 질문에 대해 Agent 실행 중...")
    
    for i, q in enumerate(questions):
        print(f"  - [{i+1}/{len(questions)}] 질문: {q[:30]}...")
        try:
            # Agent 실행 (Thread ID를 분리하여 독립성 보장)
            result = await agent_executor.ainvoke(
                {"messages": [("user", q)]},
                config={"configurable": {"thread_id": f"eval_{project_name}_{i}"}}
            )
            
            # A. 답변 추출
            last_msg = result["messages"][-1]
            answers.append(last_msg.content)
            
            # B. Context 추출 (ToolMessage parsing)
            # 우리 시스템의 Tool은 JSON string을 반환하므로 파싱 필요
            retrieved_ctx = []
            for msg in result["messages"]:
                if msg.type == "tool":
                    try:
                        # tool output이 json string인 경우
                        content_dict = json.loads(msg.content)
                        if "context" in content_dict:
                            # context가 긴 문자열 하나로 되어있으므로 리스트에 담음
                            retrieved_ctx.append(content_dict["context"])
                    except:
                        # json이 아니거나 실패하면 raw content 사용 (fallback)
                        retrieved_ctx.append(str(msg.content))
            
            # 검색된게 없으면 빈 문자열 처리
            contexts_list.append(retrieved_ctx if retrieved_ctx else [""])
            
        except Exception as e:
            print(f"    ⚠️ Agent 실행 에러: {e}")
            answers.append("Error occurred")
            contexts_list.append([""])

    # 3. RAGAS Dataset 구성
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    ragas_dataset = Dataset.from_dict(data_dict)
    
    # 4. 평가 모델 설정
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # 랭체인 모델 생성
    judge_llm_raw = JSONCleanLLM(model="gpt-4o", temperature=0)
    creative_llm_raw = JSONCleanLLM(model="gpt-4o", temperature=0.7)
    embeddings_raw = OpenAIEmbeddings(model="text-embedding-3-large")

    # RAGAS 최신 버전 규격에 맞게 Wrapper로 감싸기
    judge_llm = LangchainLLMWrapper(judge_llm_raw)
    creative_llm = LangchainLLMWrapper(creative_llm_raw)
    embeddings = LangchainEmbeddingsWrapper(embeddings_raw)
    
    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
        AnswerRelevancy(llm=creative_llm, embeddings=embeddings),
        AnswerCorrectness(llm=judge_llm, embeddings=embeddings)
    ]
    
    # 5. RAGAS 실행
    print("⚖️ RAGAS Metrics 계산 중...")
    results = ragas_evaluate(
        ragas_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings
    )
    
    # 6. 결과 저장
    df_result = results.to_pandas()
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 평가 완료! 결과 저장됨: {output_file}")
    
    return results
