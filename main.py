from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests
import asyncio
import aiohttp
import os
from typing import List
from openai import OpenAI, AsyncOpenAI

app = FastAPI()

your_api_key = os.getenv("openai_api_key")

class TextRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    queries: List[str]

class VerificationRequest(BaseModel):
    texts: List[str]
    docs: List[str]

class PipelineRequest(BaseModel):
    text: str

@app.post("/extract_uf")
def gpt_unitfact_extraction(request: TextRequest):
    text = request.text
    
    def _call_gpt(text: str) -> str:
        client = OpenAI(api_key=your_api_key)
        responses = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
            給定一段包含單位事實（Unit Fact，簡稱UF）的文本。UF是一種聲明，宣稱某事為真或假，並可由人類驗證。你的任務是準確識別並提取文本中每一個UF。
            為確保每個UF清晰無歧義，應避免使用代詞或其他指代表達，且每個UF應簡潔（少於20個字）並獨立存在。
            【UF 提取規則】：
            1. 每個 UF 必須是清晰、可獨立驗證的事實，不可包含模糊的推論。
            2. UF 應該簡潔，理想長度為 10-20 個字，最多不超過 25 個字。
            3. 避免使用代詞或指代詞（例如「他」、「她」、「這些」、「那些」等）。
            4. 每個 UF 應該是獨立的，不要產生過於相似或重複的 UF。
            5. UF 應該具體描述，不要過於抽象或概括。
            
            【回應格式】: 回應必須是字典列表形式，每個字典包含鍵「UF」和值「提取的單位事實」。你只能按照以下格式回應，不要添加任何其他內容或違反格式的註釋。
            【任務範例】:
            ［文本］：李娜在澳網決賽中直落兩盤擊敗了謝淑薇。這是她第二次贏得澳洲網球公開賽冠軍。李娜成為亞洲第一位贏得這項大滿貫的球員。
            ［回應］：
            {{"UF": "李娜在澳網決賽中直落兩盤擊敗謝淑薇"}}
            {{"UF": "李娜贏得第二個澳網冠軍"}}
            {{"UF": "李娜是第一位亞洲大滿貫得主"}}
            ［文本］：阿里巴巴宣布將投資1億元用於支持中小企業。馬雲表示，這是公司扶持創新的一部分。
            ［回應］：
            {{"UF": "阿里巴巴投資1億元支持中小企業"}}
            {{"UF": "馬雲表示投資是扶持創新的部分"}}
            ［文本］：孫士毅在其奏摺中針對潮州至贑州的文報傳遞提出了具體的建議與措施。他指出，從潮州至江西的筠門嶺再到贑州，是潮民赴江西貿易的主要通道，較之經由廣東省城的路線可以少行一千多里。因此，他建議在潮州專門派遣人員，經由筠門嶺一路傳遞文報，以加快速度。此外，他提到，從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲，因此他建議在贑縣至筠門嶺之間設立接遞文報的站點，以便更迅速地傳遞文報。這些措施旨在縮短傳遞時間，確保文報能夠及時送達。
            ［回應］：
            {{"UF": "孫士毅針對潮州至贑州的文報傳遞提出了建議。"}}
            {{"UF": "孫士毅指出潮州至江西的通道比廣東省城的路線短一千多里"}}
            {{"UF": "孫士毅建議潮州至贑州文報傳遞應經由筠門嶺"}}
            {{"UF": "孫士毅提出從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲。"}}
            {{"UF": "孫士毅建議在贑縣至筠門嶺之間設立接遞文報的站點以加速。"}}
            {{"UF": "孫士毅提出建議以縮短文報傳遞時間。"}}
            現在使用以下文本完成任務： 
            ［文本］：{text} 
            ［回應］：
            """}
            ],
        )
        return responses.choices[0].message.content
    
    def _process(ufs: str) -> List[str]:
        lines = ufs.split("\n")
        group_list = []
        for line in lines:
            try:
                data = json.loads(line)
                group_list.append(data["UF"])
            except json.JSONDecodeError:
                pass
        return group_list
    
    result = _call_gpt(text)
    return {"unit_facts": _process(result)}

@app.post("/retrieve_docs")
def retriever_api(request: QueryRequest):
    queries = request.queries
    documents = []
    
    def process(docs):
        results = ""
        for doc in docs:
            evidence = doc['page_content']
            source = doc['metadata']['source']
            results += f"參考資料:{source}\n{evidence}\n"
        return results
    
    for query in queries:
        try:
            res = requests.get(f"https://nvcenter.ntu.edu.tw:8000/retrieve?question={query}")
            raw_docs = res.json()
            documents.append(process(raw_docs))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"檢索錯誤: {e}")
            
    return {"documents": documents}


@app.post("/verify_text")
async def gpt_txt_verification(request: VerificationRequest):
    texts, docs = request.texts, request.docs
    
    async def _call_gpt(text: str, evd: str) -> str:
        client = AsyncOpenAI(api_key=your_api_key)
        responses = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位專業的評估員，並且使用繁體中文與台灣的詞彙做評估。"},
                {"role": "user", "content": f"""
                給定一段文本。你的任務是確認該文本中是否存在任何錯誤。
                當你判斷給定文本的真實性時，可以參考提供的證據，但證據不一定有幫助，且證據之間可能會相互矛盾，所以在使用證據來判斷文本的真實性時，務必小心謹慎。
                回應要是一個包含五個鍵的字典——"statement"（敘述）, "reasoning"（推理）, "error"（錯誤）, "factuality"（真實性）, "source"(來源)。
                statement不要重複，且需完全來自於文本中的文字。同時引用文本的句子時一律使用「」。
                請僅以以下格式進行回應。不要返回任何其他內容。
                [回應格式]:
                {{
                "statement": "statement",
                "reasoning": "判斷文本真實性的理由，請提供證據來支持你的判斷。",
                "error": "如果文本是正確的為 None；否則，描述文本錯誤的部分。",
                "factuality": 如果給定的文本是真實的，則為 True，否則為 False,
                "source": "資料來源，為用來判斷statement事實性的檔案或網址。如果找不到來源則顯示「無」",
                }}
                
                [範例]:
                [文本]: 京都是日本的首都，以其傳統的木造建築和千年寺廟著稱。東京塔是世界上第二高的自立式鋼塔。
                [證據]: [[{{"content": "京都是日本的歷史文化中心，曾經是日本的首都，直到1868年為止。", "source": "https://zh.wikipedia.org/zh-tw/%E4%BA%AC%E9%83%BD"}}],
                [回應]：
                {{
                "statement": "京都是日本的首都",
                "reasoning": "根據證據，京都曾是首都，但自1868年起，日本的首都已經是東京。",
                "error": "「京都是日本的首都」應更正為「京都是日本的前首都」。",
                "factuality": False,
                "source": "https://zh.wikipedia.org/zh-tw/%E4%BA%AC%E9%83%BD",
                }}
                
                現在使用以下文本和證據完成任務：
                [文本]: {text}
                [證據]: {evd}
                [回應]:
                """}
            ],
        )
        return responses.choices[0].message.content
    
    def clean_json_format(block):
        block = block.replace("'", '"').replace("False", "false").replace("True", "true").replace("None", "null")
        return block
    
    async def process_results():
        results = await asyncio.gather(*(_call_gpt(text, doc) for text, doc in zip(texts, docs)))
        return [json.loads(clean_json_format(res)) for res in results]
    
    return {"verification_results": await process_results()}

@app.post("/full_pipeline")
async def full_pipeline(request: PipelineRequest):
    text = request.text
    
    # Extract unit facts
    uf_response = gpt_unitfact_extraction(TextRequest(text=text))
    unit_facts = uf_response["unit_facts"]
    
    # Retrieve supporting documents
    doc_response = retriever_api(QueryRequest(queries=unit_facts))
    documents = doc_response["documents"]
    
    # Verify facts
    verification_response = await gpt_txt_verification(VerificationRequest(texts=unit_facts, docs=documents))
    
    return {
        "unit_facts": unit_facts,
        "documents": documents,
        "verification_results": verification_response["verification_results"]
    }
