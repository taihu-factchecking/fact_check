### This is a FastAPI server code used on the github.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests
import asyncio
import aiohttp
import os
import re
import time
from typing import List
from openai import OpenAI, AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi import Request
from datetime import datetime
from rapidfuzz import fuzz
from copy import deepcopy

app = FastAPI()

### handle CORS ###
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins= ['http://localhost:3000', 'https://hpc.psy.ntu.edu.tw/taihucais/chat'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# print("CORS middleware 已經啟用！")
### handle CORS ###

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
    docs: List[str]


# TODO: 寫一個log儲存查核的資訊
def log(message):
    return

def sentence_ngram(text):
    # 刪除中括號內的數字，例如 [1]、[23]
    text = re.sub(r'\[\d+\]', '', text)

    # 用 re.finditer 保留每句的標點
    matches = re.finditer(r'([^，。！？；]+)([，。！？；]?)', text)
    sentence_tuples = [(m.group(1).strip(), m.group(2)) for m in matches if m.group(1).strip()]

    all_ngrams = []

    for n in [1, 2, 3]:
        for i in range(len(sentence_tuples) - n + 1):
            pieces = []
            for j in range(n):
                sent, punct = sentence_tuples[i + j]
                pieces.append(sent)
                if j < n - 1:  # 中間句子之後加上它原本的標點
                    pieces.append(punct)
            ngram = ''.join(pieces).strip()  # 去除每個 ngram 的前後空白
            all_ngrams.append(ngram)

    return all_ngrams

def split_into_clauses(text):
    coarse_segments = re.split(r"[，。？！；]", text)
    fine_clauses = re.split(r"[。？！；]", text)
    results = coarse_segments + fine_clauses
    return results

def lcstring_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_len = 0
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                max_len = max(max_len, dp[i+1][j+1])
    return max_len

def lcsequence_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def find_best_match(clause, candidates):
    best_score = 0
    best_clause = ""
    for c in candidates:
        score = lcstring_len(clause, c)
        if score > best_score:
            best_score = score
            best_clause = c
    return best_clause

def _find_span(clause: str, full_text: str):
    sentences = split_into_clauses(full_text)
    best_match = find_best_match(clause, sentences)

    # TODO: 要處理中括號出現在句子中的錯誤
    best_match = re.sub(r"\[.*?\]|\n", "", best_match)  # 預設中括號只會出現在句子結尾
    start = full_text.find(best_match)
    end = start + len(best_match)
    return (start, end)

def _find_span_fuzzy(clause, full_text, seg):
    segments = seg
    best_score = 0
    best_segment = ""
    for segment in segments:
        score = fuzz.ratio(clause, segment)
        if score > best_score:
            best_score = score
            best_segment = segment
    
    # 從 full_text 找出 best_segment 的實際 span
    start = full_text.find(best_segment)
    end = start + len(best_segment)
    print((clause, full_text[start:end]))
    return (start, end), best_segment

def _present(text, results):
    idx_evidence_map = {tuple(r["idx"]): r["evidence"] for r in results["verification_results"]}
    idx_list = sorted(idx_evidence_map.keys())

    output = []
    prev = 0
    for start, end in idx_list:
        if prev < start:
            output.append(text[prev:start])  # 非 evidence 區段
        snippet = text[start:end + 1]
        evidence = idx_evidence_map[(start, end)]
        output.append(f"{snippet} ← {evidence}")  # 加上對應的 evidence
        prev = end + 1
    if prev < len(text):
        output.append(text[prev:])  # 剩下的部分

    return "\n".join(output)

def _merge_verification_results(results):
    from copy import deepcopy

    merged = []
    segments = deepcopy(results["verification_results"])
    if not segments:
        return {"verification_results": []}

    used = [False] * len(segments)

    for s in segments:
        s["start"] = s["idx"][0]
        s["end"] = s["idx"][1]
        s["claim"] = [s["claim"]] if not isinstance(s["claim"], list) else s["claim"]
        s["reasoning"] = [s["reasoning"]] if not isinstance(s["reasoning"], list) else s["reasoning"]

    segments_sorted = sorted(enumerate(segments), key=lambda x: x[1]["start"])

    for i, seg in segments_sorted:
        if used[i]:
            continue
        used[i] = True
        current = deepcopy(seg)

        for j in range(i + 1, len(segments_sorted)):
            k, next_seg = segments_sorted[j]
            if used[k]:
                continue

            # 放寬 evidence 比對條件：子字串即可
            evidence_match = (
                current.get("evidence") in next_seg.get("evidence") or
                next_seg.get("evidence") in current.get("evidence")
            )

            filename_match = (
                current.get("filename") in next_seg.get("filename") or
                next_seg.get("filename") in current.get("filename")
            )

            other_keys_match = all(
                current.get(key) == next_seg.get(key)
                for key in ["docid", "data_source", "data_title", "data_date"]
            )

            if (
                current["factuality"] == next_seg["factuality"] and
                current["end"] + 7 >= next_seg["start"] and
                other_keys_match and
                evidence_match and 
                filename_match
            ):
                current["end"] = next_seg["end"]
                current["idx"] = [current["start"], current["end"]]
                current["claim"].extend(next_seg["claim"] if isinstance(next_seg["claim"], list) else [next_seg["claim"]])
                current["reasoning"].extend(next_seg["reasoning"] if isinstance(next_seg["reasoning"], list) else [next_seg["reasoning"]])
                used[k] = True
            else:
                break

        current.pop("start", None)
        current.pop("end", None)
        merged.append(current)

    return {"verification_results": merged}

def _remove_substrings(cls):
    to_remove = set()
    for i in range(len(cls)):
        for j in range(len(cls)):
            if i != j and cls[i] in cls[j]:
                to_remove.add(cls[i])
    return [s for s in cls if s not in to_remove]


@app.post("/extract_uf")
async def claim_extraction(request: TextRequest):
    text = request.text
    
    async def _call_gpt(text: str) -> str:
        client = AsyncOpenAI(api_key=your_api_key)
        responses = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
給定一段文本，你的任務是準確識別並依序提取文本中每一個聲明（claim），並標註聲明對應到文本的原始句子（可為一或多句）。聲明是一段事實的敘述，並可由人類驗證為真或假。為確保每個聲明清晰無歧義，應避免使用代詞或其他指代表達。
【聲明提取規則】：
1. 每個聲明必須是清晰、可獨立驗證的事實，不可包含模糊的推論。
2. 聲明須為完整的事實，包含其在文本中的上下文背景。
3. 避免使用代詞或指代詞（例如「他」、「她」、「這些」、「那些」等）。
4. 每個聲明應該是獨立的，不要產生過於相似或重複的聲明。
5. 聲明應該具體描述，不要過於抽象或概括。
【回應格式】：回應必須是字典列表形式，每個字典包含兩個鍵：「claim」：提取的單位事實；「clause」：claim對應到文本的子句（可包含一或多句），代表claim是從文本的那些子句中提取出事實的，一定要與文本的某個子字串相符，不要把不相關的子句包含進來，且clause之間不要有重疊的部分；此外，如果子句的前面有連接詞，且包含進來不影響文義，則一併包含進來。
你只能按照以下格式回應，不要添加任何其他內容或違反格式的註釋。
【任務範例】：
［文本］：李娜在澳網決賽中，直落兩盤擊敗了謝淑薇，這是她第二次贏得了澳洲網球公開賽冠軍，非常厲害。在這場比賽之後，李娜成為亞洲第一位贏得這項大滿貫的球員。
［回應］：
{{"claim": "李娜在澳網決賽中直落兩盤擊敗謝淑薇", "clause": "李娜在澳網決賽中直落兩盤擊敗了謝淑薇"}}
{{"claim": "李娜贏得第二個澳網冠軍", "clause": "這是她第二次贏得了澳洲網球公開賽冠軍"}}
{{"claim": "李娜是第一位亞洲大滿貫得主", "clause": "在這場比賽之後，李娜成為亞洲第一位贏得這項大滿貫的球員"}}

［文本］：阿里巴巴宣布將投資1億元用於支持中小企業。馬雲表示，這是公司扶持創新的一部分。
［回應］：
{{"claim": "阿里巴巴投資1億元支持中小企業", "clause": "阿里巴巴宣布將投資1億元用於支持中小企業"}}
{{"claim": "馬雲表示投資是扶持創新的部分", "clause": "馬雲表示，這是公司扶持創新的一部分"}}

［文本］：孫士毅在其奏摺中針對潮州至贑州的文報傳遞提出了具體的建議與措施。他指出，從潮州至江西的筠門嶺再到贑州，是潮民赴江西貿易的主要通道，較之經由廣東省城的路線可以少行一千多里。因此，他建議在潮州專門派遣人員，經由筠門嶺一路傳遞文報，以加快速度。此外，他提到，從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲，因此他建議在贑縣至筠門嶺之間設立接遞文報的站點，以便更迅速地傳遞文報。這些措施旨在縮短傳遞時間，確保文報能夠及時送達。
［回應］：
{{"claim": "孫士毅針對潮州至贑州的文報傳遞提出了建議。", "clause": "孫士毅在其奏摺中針對潮州至贑州的文報傳遞提出了具體的建議與措施"}}
{{"claim": "孫士毅指出潮州至江西的通道比廣東省城的路線短一千多里", "clause": "他指出，從潮州至江西的筠門嶺再到贑州，是潮民赴江西貿易的主要通道，較之經由廣東省城的路線可以少行一千多里"}}
{{"claim": "孫士毅建議潮州至贑州文報傳遞應經由筠門嶺", "clause": "因此，他建議在潮州專門派遣人員，經由筠門嶺一路傳遞文報"}}
{{"claim": "孫士毅提出從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲。", "clause": "此外，他提到，從贑州到潮州的文報傳遞若仍經由廣東省城，會導致延遲"}}
{{"claim": "孫士毅建議在贑縣至筠門嶺之間設立接遞文報的站點以加速。", "clause": "因此他建議在贑縣至筠門嶺之間設立接遞文報的站點，以便更迅速地傳遞文報"}}
{{"claim": "孫士毅提出建議以縮短文報傳遞時間。", "clause": "這些措施旨在縮短傳遞時間，確保文報能夠及時送達"}}

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
    

    result = await _call_gpt(text)
    text = re.sub(r"\[.*?\]", "", text)  # 去除中括號及其內容
    cl2_pair = _process(result)  # 將("claim", "clause")拆分出來
    clm_list = [pair[0] for pair in cl2_pair]  # 提取出所有聲明
    cls_list = [pair[1] for pair in cl2_pair]  # 提取出所有原句
    # cl_index = [(text.index(pair[1]), text.index(pair[1])+len(pair[1])) for pair in cl2_pair]
    # cl_index = [_find_span(pair[1], text) for pair in cl2_pair]

    return {"claims": clm_list, "clauses": cls_list}

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
回應要是一個包含五個鍵的字典——"claim"（敘述）, "factuality"（真實性）, "filename" (檔案名稱), "evidence" (證據片段), "reasoning" (理由)。
請使用以下格式進行回應。不要返回任何其他內容。
[回應格式]:
{{
"claim": "提取的聲明",
"reasoning": "判斷claim是否被evidence支持的原因",
"factuality": 如果claim有被證據支持則為True，如果claim被證據反駁或claim跟證據無關為False,
"filename": "如果證據跟claim沒有關聯，則不要放任何檔案的名稱，放None；如果claim有被證據支持或反駁，則放檔案**全名**",
"evidence": "用以判斷claim真實性的證據片段，應回傳包含證據片段的整個段落。如果claim沒有被任何證據支持，則為None"
}}
                
[範例]
[文本]: 京都是日本的歷史文化中心
[證據]: "檔案1:\n臣孫士毅跪奏。再，臣遵旨赴潮駐劄，所有奏報事件，若仍由廣東省城前至江西，道路紆迴，不免稽延時日。查自潮州至江西之筠門嶺前赴贑州，為潮民赴江西貿易往來孔道，較之經由廣東省城可少行一千幾百里。是以臣在潮專差齎摺進京，俱走筠門嶺一路，即由驛拜發各摺，亦俱自行雇備人夫，專差齎至贑縣，再行發遞，以期迅速。惟由贑來潮，遇有馳遞諭旨，若仍從廣東省城轉遞，計期太遲。當將自贑縣至筠門嶺可否接遞文報之處，咨商江西撫臣何裕城。茲准覆稱，現已量為酌撥。臣上次接奉諭旨，亦已由筠門嶺遞到，應請俟臺逆將次事竣，粵省無甚交涉事件，即行知會江西，撤去腰站。合即附片聲明。謹奏。〔硃批〕：好，知道了。\n\n檔案2:孫士毅於乾隆五十二年四月二十四日對於文報傳遞路徑的調整，具體提出了一些改道方案。"
[回應]：
{{
"claim": "京都是日本的歷史文化中心",
"reasoning": "提供的證據與京都是否為日本的歷史文化中心無直接關聯。",
"factuality": False,           
"filename": None,
"evidence": None
}}
                 
[範例]
[文本]: 改道方案使用潮州經筠門嶺到贑州路線
[證據]: "檔案1:\n臣孫士毅跪奏。再，臣遵旨赴潮駐劄，所有奏報事件，若仍由廣東省城前至江西，道路紆迴，不免稽延時日。查自潮州至江西之筠門嶺前赴贑州，為潮民赴江西貿易往來孔道，較之經由廣東省城可少行一千幾百里。是以臣在潮專差齎摺進京，俱走筠門嶺一路，即由驛拜發各摺，亦俱自行雇備人夫，專差齎至贑縣，再行發遞，以期迅速。惟由贑來潮，遇有馳遞諭旨，若仍從廣東省城轉遞，計期太遲。當將自贑縣至筠門嶺可否接遞文報之處，咨商江西撫臣何裕城。茲准覆稱，現已量為酌撥。臣上次接奉諭旨，亦已由筠門嶺遞到，應請俟臺逆將次事竣，粵省無甚交涉事件，即行知會江西，撤去腰站。合即附片聲明。謹奏。〔硃批〕：好，知道了。\n\n檔案2:京都曾經是日本的首都，直到1868年為止。京都富有濃厚的文化氣息。"
[回應]：
{{
"claim": "改道方案使用潮州經筠門嶺到贑州路線",
"reasoning": "由證據顯示，改道方案使用自潮州至江西之筠門嶺前赴贑州的往來通道。",
"factuality": True,
"filename": "檔案1",
"evidence": "查自潮州至江西之筠門嶺前赴贑州，為潮民赴江西貿易往來孔道，較之經由廣東省城可少行一千幾百里。"
}}
                 
[範例]
[文本]: 一元化課程對升學生不適用
[證據]: "檔案1:\n省議會關切教育體系的公平性和多元性\n\n檔案2:本省與北、高二市所受教育的質差很多，不知感想如何？延長十二年國教　對學生數的如何分配與現行教育體系下是否足夠分配，均應慎重考慮；目前課程的僵化是值得檢討，尤其課程的一元化是適應於升學的，對於不升學的根本無法接受；如教材要有所改進時，建議應要如何落實本土化教育；如果調整教材之後老師是否要進修？對於新教材要有新的教法？"
[回應]：
{{
"claim": "一元化課程對升學生不適用",
"reasoning": "由證據顯示，一元化課程適用於升學生",
"factuality": False,
"filename": "檔案2",
"evidence": "目前課程的僵化是值得檢討，尤其課程的一元化是適應於升學的，對於不升學的根本無法接受；"
}}

                
現在使用以下證據搭配文本完成任務：
[文本]: {text}
[證據]: {evd}
[回應]:"""}
            ],
        )
        return responses.choices[0].message.content
    
    def clean_json_format(block):
        block = block.replace("'", '\\"').replace("False", "false").replace("True", "true").replace("None", "null")
        return block
    
    async def process_results():
        try:
            results = await asyncio.gather(*(_call_gpt(text, doc) for text, doc in zip(texts, docs)))
            return [json.loads(clean_json_format(res)) for res in results]
        
        except json.JSONDecodeError as e:
            print("錯誤: ", results)
        
    
    return {"verification_results": await process_results()}

@app.post("/fact_check_pipeline")
async def full_pipeline(request: PipelineRequest):
    print("Execution Time:　", datetime.now())
    execution_time_start = time.time()
    text = request.text  # String
    input_docs = request.docs  # List[String] e.g. ['TPC03GAZ-00211314_1763_1.txt', 'TPC03GAZ-00052484_0163_1.txt']
    docs = []
    data_from = ""
    print("text: ", text)
    print("input_docs: ", input_docs)

    print("process input...")
    ### Process input_data to the format ###
    for doc in input_docs:
        if "TPC" in doc:  # 省議會公報
            data_from = "省議會公報"
            _fulltext = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/full_text?title={doc}""")  # 取得全文
            _fulltext = _fulltext.json()
            _metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={doc}""")  # 取得metadata
            _metadata = _metadata.json()
            try:
                # page_content = _metadata["data_title"] + "。概要：" + _metadata["abstract"]  # 證據內容，由檔名與摘要組成
                page_content = _fulltext["full_text"]
                metadata = {  # 這裡放前端需要的metadata
                    "source": doc
                    }
                other = {  # 這裡放處理過程中需要的metadata
                    "id": _metadata["identifier"],  # id e.g. "003-02-03OA-05-6-4-00-00272"
                    "data_from": data_from,
                    "date_string": _metadata["date_string"]
                }
            except KeyError as e:
                print("Exception: ", e)
                print("metadata: ", _metadata)
            

        else:  # 明清台灣行政檔案
            # doc = "".join(doc[:doc.rindex("-")])
            data_from = "明清台灣行政檔案"
            _fulltext = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/full_text?title={doc}""")  # 取得全文
            _fulltext = _fulltext.json()
            _metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={"".join(doc[:doc.rindex("-")])}""")
            _metadata = _metadata.json()
            page_content = _fulltext["full_text"]
            metadata = {  # 這裡放前端需要的metadata
                "source": doc
                }
            other = {  # 這裡放處理過程中需要的metadata
                "data_from": data_from,
                "date_string": _metadata["date_string"]
            }

        doc_info = {
            "page_content": page_content,
            "metadata": metadata,
            "other": other
        }
        docs.append(doc_info)
    ########################################

    
    clm_cls = {}  # claim-clause pairs | Key: claim, Values: clause
    
    ### Prepare claims ###
    execution_time_process_input = time.time()
    print("extract claims...")
    raw_claims = await claim_extraction(TextRequest(text=text))  # Dictionary{claims: List[String], clause: List[String]}
    for claim, clause in zip(raw_claims["claims"], raw_claims["clauses"]):
        clm_cls[claim] = clause
    claims = raw_claims["claims"]
    clauses = raw_claims["clauses"]
    ######################
    

    ### Prepare documents ###
    doc_text = ""  # 將所有的證據組成一個字串
    doc_id = {None: None}
    for doc in docs: # docs: 所有證據的相關資訊
        doc_text += f"""檔案名稱：{doc["metadata"]["source"]}\n檔案時間：{doc["other"]["date_string"]}\n檔案內容{doc["page_content"]}\n\n"""

        # 省議會公報(有id)
        if "id" in doc["other"]:
            doc_id[doc["metadata"]["source"]] = doc["other"]["id"]  # 將id資訊存起來，Dictionary{filename: file_id}
        # 明清台灣行政檔案(沒有id)
        else:
            continue
    ######################### 


    ### Verify facts ###
    execution_time_extract_claims = time.time()
    print("verify claims...")
    verification_response = await claim_verification(VerificationRequest(texts=_remove_substrings(clauses), docs=[doc_text]*len(claims)))
    ''' Format (verification_response)
    {
    "claim": String
    "factuality": True or False
    "source": String | Filename or None 
    "evidence": String
    "reasoning": String
    }
    '''

    ''' 要輸出給前端的格式
    {
    "claim": String,
    "factuality": True or False,
    "filename": String | File name or None,
    "evidence": String,
    "idx": List(Int), len(List) = 2,
    "docid": String,
    "url": String,
    "data_source": String,
    "data_title": String,
    "data_date": String,
    }
    '''

    execution_time_verify_claims = time.time()
    # text_segments = split_into_clauses(text)

    print(verification_response["verification_results"])
    # print(clm_cls)

    seg = sentence_ngram(text)
    for res_idx, result in enumerate(verification_response["verification_results"]):       
        # cls = clm_cls[claims[res_idx]]  # 找到clause
        cls = result["claim"]
        if cls[-1] in "，。？！；":
            cls = cls[:-1].replace(" ", "")
        cls_info = _find_span_fuzzy(cls, text, seg) 
        result["idx"] = cls_info[0]
        cls = cls_info[1]
        # print(cls)
        # result["idx"] = (text.index(cls), text.index(cls)+len(cls))


        for input_doc in input_docs:
            # print("input_doc:", input_doc)
            # print("result['filename']:", result["filename"])
            if result["filename"] and (result["filename"] in input_doc or input_doc in result["filename"]):
                result["filename"] = input_doc
                break


        # result["idx"], best_match = _find_span_fuzzy(cls, text, text_segments)  # 新增欄位"idx"到result裡面，表示對應到output的位置資訊
        # best_match_sentence = best_match.split("，")
        # text_segments = [s for s in text_segments if not any(sub in s for sub in best_match_sentence)]
        input_file = result["filename"]
        if input_file not in [None, 'null']:
            if input_file.index("-")!=input_file.rindex("-"):
                input_file = "".join(result["filename"][:result["filename"].rindex("-")])
            metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={input_file}""")
            metadata = metadata.json()
        
        # 省議會公報 如果factuality=True
        if result["filename"] not in [None, 'null'] and metadata["collect_name"]=="臺灣省議會公報": # and result["filename"] in doc_id:
            
            # result["docid"] = doc_id[result["filename"]]  # 新增欄位"docid"到result裡面
            
            metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={result["filename"]}""")
            metadata = metadata.json()
            result["url"] = metadata["source_url"]


            # 新增 1.來源 2.標題 3.日期 4.證據句子 到輸出
                
            result["data_source"] = metadata["collect_name"]
            result["data_title"] = metadata["data_title"]
            result["data_date"] = metadata["date_string"]
            ##########################################

        # 明清台灣行政檔案
        elif result["filename"] not in [None, 'null'] and (metadata["collect_name"]=="明清檔案" or metadata["collect_name"]=="漢文台灣日日新報"):
            input_file = result["filename"]
            if input_file.index("-")!=input_file.rindex("-"):
                input_file = "".join(result["filename"][:result["filename"].rindex("-")])

            result["docid"] = ""  # 明清台灣行政檔案沒有identifier
            result["url"] = f"""https://nvcenter.ntu.edu.tw:8000/full_text?title={result["filename"]}"""  # 獲得全文


            # 新增 1.來源 2.標題 3.日期 4.證據句子 到輸出
            ##########################################
            metadata = requests.get(f"""https://nvcenter.ntu.edu.tw:8000/metadata?title={input_file}""")
            metadata = metadata.json()

            result["data_source"] = metadata["collect_name"]
            result["data_title"] = metadata["abstract"]
            result["data_date"] = metadata["date_string"]
            ##########################################
        
        else:  # 證據不相關
            result["docid"] = ""
            result["url"] = ""
            result["data_source"] = "無證據支持"
            result["data_title"] = ""
            result["data_date"] = ""

    
    ####################


    print("############### Results ###############")
    print()
    for result in verification_response["verification_results"]:
        print(result)
    print()
    print("############### Results ###############")
    
    print()
    # print("[DEBUG] GPT 回傳：", result)

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    execution_time_end = time.time()
    execution_time = execution_time_end - execution_time_start

    print("############### Execution Time ###############")
    print()
    print("process input:", execution_time_process_input-execution_time_start)
    print("extract claims:", execution_time_extract_claims-execution_time_process_input)
    print("verify claims:", execution_time_verify_claims-execution_time_extract_claims)
    print("present result:", execution_time_end-execution_time_verify_claims)
    print("Total Execution time:", execution_time)
    print()
    print("############### Execution Time ###############")

    print()


    try:
        new_results = _merge_verification_results({"verification_results": verification_response["verification_results"]})
        ### presentation ###
        print("############### Presentation ###############")
        print()
        print(_present(text, new_results))
        print()
        print("############### Presentation ###############")
        ####################
        if not new_results:
            return {"verification_results": verification_response["verification_results"]}
        return new_results
    except Exception:
        return {"verification_results": verification_response["verification_results"]}
    
