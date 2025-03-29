### This is a FastAPI server code used on the github.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests
import asyncio
import aiohttp
import os
import re
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
    docs: List[dict]

@app.post("/extract_uf")
def claim_extraction(request: TextRequest):
    text = request.text
    
    def _call_gpt(text: str) -> str:
        client = OpenAI(api_key=your_api_key)
        responses = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
給定一段包含聲明（claim）的文本，聲明宣稱某敘述為真或假，並可由人類驗證。你的任務是準確識別並提取文本中每一個聲明，並標註聲明對應到文本的原始句子（可為一或多句）。為確保每個聲明清晰無歧義，應避免使用代詞或其他指代表達，且每個聲明應簡潔並獨立存在。
【聲明提取規則】：
1. 每個聲明必須是清晰、可獨立驗證的事實，不可包含模糊的推論。
2. 聲明應該簡潔，理想長度為 10-20 個字，最多不超過 25 個字。
3. 避免使用代詞或指代詞（例如「他」、「她」、「這些」、「那些」等）。
4. 每個聲明應該是獨立的，不要產生過於相似或重複的聲明。
5. 聲明應該具體描述，不要過於抽象或概括。
【回應格式】：回應必須是字典列表形式，每個字典包含兩個鍵：「claim」：提取的單位事實；「clause」：聲明對應的原始子句（可包含一或多句），請盡量精準，不要把不相關的子句包含進來。
你只能按照以下格式回應，不要添加任何其他內容或違反格式的註釋。
【任務範例】：
［文本］：李娜在澳網決賽中，直落兩盤擊敗了謝淑薇。這是她第二次贏得了澳洲網球公開賽冠軍，非常厲害。在這場比賽之後，李娜成為亞洲第一位贏得這項大滿貫的球員。
［回應］：
{{"claim": "李娜在澳網決賽中直落兩盤擊敗謝淑薇", "clause": "李娜在澳網決賽中直落兩盤擊敗了謝淑薇"}}
{{"claim": "李娜贏得第二個澳網冠軍", "clause": "這是她第二次贏得了澳洲網球公開賽冠軍"}}
{{"claim": "李娜是第一位亞洲大滿貫得主", "clause": "李娜成為亞洲第一位贏得這項大滿貫的球員"}}

［文本］：阿里巴巴宣布將投資1億元用於支持中小企業。馬雲表示，這是公司扶持創新的一部分。
［回應］：
{{"claim": "阿里巴巴投資1億元支持中小企業", "clause": "阿里巴巴宣布將投資1億元用於支持中小企業"}}
{{"claim": "馬雲表示投資是扶持創新的部分", "clause": "馬雲表示，這是公司扶持創新的一部分"}}

［文本］：孫士毅在其奏摺中針對潮州至贑州的文報傳遞提出了具體的建議與措施。他指出，從潮州至江西的筠門嶺再到贑州，是潮民赴江西貿易的主要通道，較之經由廣東省城的路線可以少行一千多里。因此，他建議在潮州專門派遣人員，經由筠門嶺一路傳遞文報，以加快速度。此外，他提到，從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲，因此他建議在贑縣至筠門嶺之間設立接遞文報的站點，以便更迅速地傳遞文報。這些措施旨在縮短傳遞時間，確保文報能夠及時送達。
［回應］：
{{"claim": "孫士毅針對潮州至贑州的文報傳遞提出了建議。", "clause": "孫士毅在其奏摺中針對潮州至贑州的文報傳遞提出了具體的建議與措施"}}
{{"claim": "孫士毅指出潮州至江西的通道比廣東省城的路線短一千多里", "clause": "他指出，從潮州至江西的筠門嶺再到贑州，是潮民赴江西貿易的主要通道，較之經由廣東省城的路線可以少行一千多里"}}
{{"claim": "孫士毅建議潮州至贑州文報傳遞應經由筠門嶺", "clause": "他建議在潮州專門派遣人員，經由筠門嶺一路傳遞文報"}}
{{"claim": "孫士毅提出從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲。", "clause": "他提到，從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲"}}
{{"claim": "孫士毅建議在贑縣至筠門嶺之間設立接遞文報的站點以加速。", "clause": "因此他建議在贑縣至筠門嶺之間設立接遞文報的站點，以便更迅速地傳遞文報"}}
{{"claim": "孫士毅提出建議以縮短文報傳遞時間。", "clause": "這些措施旨在縮短傳遞時間"}}

現在使用以下文本完成任務：
［文本］：{text}
［回應］："""}
            ],
        )
        return responses.choices[0].message.content
    
    def _process(ufs: str) -> List[str]:
        lines = ufs.split("\n")
        group_list = []
        for line in lines:
            try:
                data = json.loads(line)
                group_list.append((data.get("claim", ""), data.get("clause", "")))
            except json.JSONDecodeError:
                pass
        return group_list
    
    result = _call_gpt(text)
    text = re.sub(r"\[.*?\]", "", text)
    cl2_pair = _process(result)
    cl_list = [pair[0] for pair in cl2_pair]
    # print(cl_list)
    cl_index = [(text.index(pair[1]), text.index(pair[1])+len(pair[1])) for pair in cl2_pair]
    return {"claims": cl_list, "index": cl_index}

@app.post("/retrieve_docs")
def retriever_api(request: QueryRequest):
    queries = request.queries
    documents = []
    ids = []
    
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
            ids.append([doc["other"]["id"] for doc in raw_docs])
            documents.append(process(raw_docs))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"檢索錯誤: {e}")
            
    return {"documents": documents, "ids": ids}


@app.post("/verify_text")
async def claim_verification(request: VerificationRequest):
    texts, docs = request.texts, request.docs
    
    async def _call_gpt(text: str, evd: str) -> str:
        client = AsyncOpenAI(api_key=your_api_key)
        responses = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位專業的評估員，並且使用繁體中文與台灣的詞彙做評估。"},
                {"role": "user", "content": f"""
給定聲明及證據。你的任務是確認該聲明是否能被證據支持。
當你判斷給定聲明的真實性時，可以參考提供的證據，但證據不一定有幫助，且證據之間可能會相互矛盾，所以在使用證據來判斷聲明的真實性時，務必小心謹慎。
回應要是一個包含三個鍵的字典——"claim"（敘述）, "factuality"（真實性）, "source"(來源)。
請使用以下格式進行回應。不要返回任何其他內容。
[回應格式]:
{{
"claim": "提取的聲明",
"factuality": 如果claim有被證據支持則為 True，否則為 False,
"source": "資料來源。如果factuality為True，則為支持claim的檔案名稱。否則為None",
}}
                
[範例]:
[文本]: 京都是日本的首都，以其傳統的木造建築和千年寺廟著稱。東京塔是世界上第二高的自立式鋼塔。
[證據]: "檔案1:\n京都是日本的歷史文化中心。\n\n檔案2:京都曾經是日本的首都，直到1868年為止。",
[回應]：
{{
"claim": "京都是日本的首都",
"factuality": False,
"source": "檔案2"
}}
                
現在使用以下文本和證據完成任務：
[文本]: {text}
[證據]: {evd}
[回應]:"""}
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

@app.post("/fact_check_pipeline")
async def full_pipeline(request: PipelineRequest):
    text = request.text
    docs = request.docs
    
    claim_idx = {}
    
    # Prepare claims
    raw_claims = claim_extraction(TextRequest(text=text))
    for claim, idx in zip(raw_claims["claims"], raw_claims["index"]):
        claim_idx[claim] = idx
    claims = raw_claims["claims"]
    
    # Prepare documents
    doc_text = ""
    doc_id = {}
    for doc in docs:
        doc_text += f"""{doc["metadata"]["source"]}:\n{doc["page_content"]}\n\n"""
        # 省議會公報
        if "id" in doc["other"]:
            doc_id[doc["metadata"]["source"]] = doc["other"]["id"]
        # 明清台灣行政檔案
        else:
            continue

    
    # Verify facts
    verification_response = await claim_verification(VerificationRequest(texts=claims, docs=[doc_text]*len(claims)))

    for result in verification_response["verification_results"]:
        result["idx"] = claim_idx[result["claim"]]
        # 省議會公報
        if result["source"] in doc_id:
            result["docid"] = doc_id[result["source"]]
            if result["source"] != None:
                metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={result["source"]}""")
                metadata = metadata.json()
                print(metadata)
                result["url"] = metadata["source_url"]
            else:
                result["url"] = None
        # 明清台灣行政檔案
        else:
            result["docid"] = ""
            result["url"] = f"""https://nvcenter.ntu.edu.tw:8000/full_text?title={result["source"]}"""
    
    
    return {
        "verification_results": verification_response["verification_results"]
    }

