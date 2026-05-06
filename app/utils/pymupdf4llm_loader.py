import os
import re
import base64
import time
from typing import Iterator, Optional, Literal, Union, List
from concurrent.futures import ThreadPoolExecutor

import pymupdf4llm
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from app.utils.message_utils import normalize_content


'''
1. 텍스트만 필요할 때 (가장 빠름)

    loader = PyMuPDF4LLMLoader("doc.pdf", extract_images=False)
    docs = loader.load()

2. 텍스트 추출과 함께 이미지 저장
    #이미지가 ./extracted_images 폴더에 png로 저장됩니다.
    #마크다운 텍스트에는 ![이미지](./extracted_images/img1.png) 링크가 남습니다.
    #LLM 비용이 들지 않습니다.

    loader = PyMuPDF4LLMLoader(
        "doc.pdf", 
        extract_images=True, 
        model=None  # 모델을 안 넣으면 추출만 함
    )
    docs = loader.load()

3. 이미지 저장 + AI 분석까지 : VLM이 내용을 보고 이미지 위치에 대체 텍스트로 추가
    gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) # Langchain ChatModel
    loader = PyMuPDF4LLMLoader(
        "doc.pdf", 
        extract_images=True, 
        model=gemini  # 모델을 넣으면 분석까지 함
    )
    docs = loader.load()
'''

class PyMuPDF4LLMLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        mode: Literal["page", "single"] = "page",
        extract_images: bool = False,
        model: Optional[BaseChatModel] = None,
        image_output_dir: str = "extracted_images",
        max_workers: int = 8  # 병렬 워커 수
    ):
        self.file_path = file_path
        self.mode = mode
        self.extract_images = extract_images
        self.model = model
        self.image_output_dir = image_output_dir
        self.max_workers = max_workers

    def lazy_load(self) -> Iterator[Document]:
        if self.extract_images:
            os.makedirs(self.image_output_dir, exist_ok=True)
            
            # action 변수 설정 (병렬 워커 수 정보 포함)
            base_action = "Analysis" if self.model else "Extraction Only"
            action = f"{base_action} (Parallel x{self.max_workers})"
            
            # ✅ 요청하신 포맷 적용
            print(f"📂 [Loader] PDF 로드: {self.file_path} (Images: {action}, Mode: {self.mode})")
        
        # 1. CPU 파싱 (순차)
        raw_output = pymupdf4llm.to_markdown(
            doc=self.file_path,
            page_chunks=(self.mode == "page"),
            write_images=self.extract_images,
            image_path=self.image_output_dir if self.extract_images else None,
            image_format="png",
            dpi=300,
            force_text=True
        )

        # 2. 결과 처리 (병렬)
        if self.mode == "page":
            yield from self._process_page_mode_parallel(raw_output)
        else:
            yield from self._process_single_mode(raw_output)

    def _process_page_mode_parallel(self, raw_pages: list) -> Iterator[Document]:
        total_pages = len(raw_pages)
        print(f"   🚀 {total_pages}개 페이지 병렬 분석 시작...")
        
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            for i, page_data in enumerate(raw_pages):
                future = executor.submit(self._process_single_page_task, page_data, i)
                futures.append(future)
            
            # 결과 수집 (순서 보장)
            for i, future in enumerate(futures):
                try:
                    doc = future.result() # 여기서 완료될 때까지 대기
                    
                    # 진행 상황 출력 (tqdm 대체)
                    # 예: [Loader] Progress: 3/10 (30.0%) 완료
                    progress = i + 1
                    percent = (progress / total_pages) * 100
                    print(f"   ⏳ [Progress] {progress}/{total_pages} ({percent:.1f}%) - Page {doc.metadata.get('page')} 완료")
                    
                    yield doc
                except Exception as e:
                    print(f"   ❌ Error processing page {i+1}: {e}")

    def _process_single_page_task(self, page_data: dict, index: int) -> Document:
        text = page_data["text"]
        meta = page_data["metadata"].copy()
        
        if 'file_path' in meta: meta['source'] = meta['file_path']
        
        # 이미지 분석 (VLM 호출)
        if self.extract_images and self.model:
            text = self._replace_images_with_captions(text, page_num=meta.get('page'))
        
        if self.extract_images:
            meta['has_images'] = bool(page_data.get('images'))
        else:
            meta['has_images'] = False

        meta.pop('images', None)
        meta.pop('tables', None)
        
        return Document(page_content=text, metadata=meta)

    def _replace_images_with_captions(self, text: str, page_num: Union[int, str]) -> str:
        image_links = re.findall(r'!\[(.*?)\]\((.*?)\)', text)
        if not image_links: return text
        
        for alt, path in image_links:
            if os.path.exists(path):
                # VLM 호출
                caption = self._analyze_image(path)
                original = f"![{alt}]({path})"
                replacement = f"{original}\n\n> **[이미지 설명]** {caption}\n"
                text = text.replace(original, replacement)
        return text

    def _analyze_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            msg = HumanMessage(content=[
                {"type": "text", "text": "Describe this image in detail for RAG context retrieval."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ])
            response = self.model.invoke([msg])
            # normalize_content: content가 str이든 list이든 안전하게 텍스트로 변환
            return normalize_content(response.content).strip()
        except Exception as e:
            return f"(Error: {e})"