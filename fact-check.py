### Packages ###
import json
import requests
import asyncio
import re
import aiohttp
import os

from typing import List
from openai import OpenAI
from openai import AsyncOpenAI
### Packages ###

your_api_key = os.getenv("openai_api_key")


def gpt_unitfact_extraction(text: str) -> List[str]:
    def _call_gpt(text: str) -> str:
        client = OpenAI(api_key=your_api_key)
        responses = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
            給定一段包含單位事實（Unit Fact，簡稱UF）的文本。UF是一種聲明，宣稱某事為真或假，並可由人類驗證。你的任務是準確識別並提取文本中每一個UF。為確保每個UF清晰無歧義，應避免使用代詞或其他指代表達，且每個UF應簡潔（少於20個字）並獨立存在。
            【回應格式】: 回應必須是字典列表形式，每個字典包含鍵「UF」和值「提取的單位事實」。你只能按照以下JSON格式回應，不要添加任何其他內容或違反格式的註釋。
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
            現在使用以下文本完成任務： 
            ［文本］：{text} 
            ［回應］：
            """}
            ],
        )
        return responses.choices[0].message.content
    
    def _process(ufs: List[str]) -> List[str]:
        lines = ufs.split("\n")  # Split each string by newlines
        group_list = []
        for line in lines:
            try:
                data = json.loads(line)
                group_list.append(data.get("UF", ""))
            except json.JSONDecodeError:
                pass  # Handle any JSON decoding errors gracefully

        return group_list
    
    result = _call_gpt(text)
    ufs = _process(result)
    return ufs


def query_expansion(facts: List[str]) -> List[str]:
    return facts


def retriever_api(questions: List[str]) -> str:
    def process(docs):
        results = ""
        for doc in docs:
            evidence = doc['page_content']
            source = doc['metadata']['source']
            results += f"參考資料:{source}\n{evidence}\n"
            
        return results
    
    documents = []
    for question in questions:
        print(question)
        try:
            res = requests.get(f"https://nvcenter.ntu.edu.tw:8000/retrieve?question={question}")
            raw_docs = res.json()
            formatted_docs = []
            for doc in raw_docs:
                formatted_doc = {
                    "page_content": doc["page_content"].replace('"', '\\"').replace("'", "\\'"),
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "chunk": doc["metadata"]["chunk"],
                        "start_idx": doc["metadata"]["start_idx"],
                    },
                }
                formatted_docs.append(formatted_doc)
            documents.append(process(formatted_docs))
        except Exception as e:
            print(f"Caught exception: {e}")
    
    return formatted_docs



def process(docs):
    results = ""
    for doc in docs:
        evidence = doc['page_content']
        source = doc['metadata']['source']
        results += f"參考資料:{source}\n{evidence}\n"
    return results


async def gpt_txt_verification(texts: List[str], docs: List[str]) -> List[str]:
    async def _call_gpt(text: str, evd: str) -> str:
        client = AsyncOpenAI(api_key=your_api_key)
        responses = await client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": "你是一位專業的評估員，並且使用繁體中文與台灣的詞彙做評估。"},
                {"role": "user", "content": f"""
                給定一段文本。你的任務是確認該文本中是否存在任何錯誤。
                當你判斷給定文本的真實性時，可以參考提供的證據，但證據不一定有幫助，且證據之間可能會相互矛盾，所以在使用證據來判斷文本的真實性時，務必小心謹慎。
                回應要是一個包含五個鍵的字典——'statement'（敘述）, 'reasoning'（推理）, 'error'（錯誤）, 'factuality'（真實性）, 'source'(來源)。
                statement不要重複，且需完全來自於文本中的文字。同時引用文本的句子時一律使用「」。
                請僅以以下格式進行回應。不要返回任何其他內容。
                [回應格式]:
                {{
                'statement': 'statement1',
                'reasoning': '判斷文本真實性的理由，請提供證據來支持你的判斷。',
                'error': '如果文本是正確的為 None；否則，描述文本錯誤的部分。',
                'factuality': 如果給定的文本是真實的，則為 True，否則為 False,
                'source': '資料來源1',
                }}
                {{
                'statement': 'statement2',
                'reasoning': '判斷文本真實性的理由，請提供證據來支持你的判斷。',
                'error': '如果文本是正確的為 None；否則，描述文本錯誤的部分。',
                'factuality': 如果給定的文本是真實的，則為 True，否則為 False,
                'source': '資料來源2',
                }}
                {{
                 ...
                }}
                
                [範例]:
                [文本]: 京都是日本的首都，以其傳統的木造建築和千年寺廟著稱。東京塔是世界上第二高的自立式鋼塔。
                [證據]: [[{{"content": "京都是日本的歷史文化中心，曾經是日本的首都，直到1868年為止。", "source": "https://zh.wikipedia.org/zh-tw/%E4%BA%AC%E9%83%BD"}}],
                        [{{"content": "東京塔高333米，是世界上第二高的自立式鋼塔，僅次於加拿大的CN塔。", "source": "https://zh.wikipedia.org/zh-tw/%E4%B8%9C%E4%BA%AC%E5%A1%94"}}]]
                [回應]：
                {{
                'statement': '京都是日本的首都',
                'reasoning': '根據證據，京都曾是首都，但自1868年起，日本的首都已經是東京。',
                'error': '「京都是日本的首都」應更正為「京都是日本的前首都」。',
                'factuality': False,
                'source': 'https://zh.wikipedia.org/zh-tw/%E4%BA%AC%E9%83%BD',
                }}
                {{
                'statement': '東京塔是世界上最高的自立式鋼塔',
                'reasoning': '根據證據，東京塔是世界上第二高的自立式鋼塔。',
                'error': None,
                'factuality': True,
                'source': 'https://zh.wikipedia.org/zh-tw/%E4%B8%9C%E4%BA%AC%E5%A1%94',
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
        # ```json
        block = re.sub(r"```json|```", "", block).strip()
        # ' -> ""
        block = block.replace('"', '\"')
        block = block.replace("'", '"')
        # boolean
        block = block.replace("False", "false").replace("True", "true").replace("None", "null")
        # , in the end
        block = re.sub(r",\s*}", "}", block)
        return block

    def process(raw_data):
        cleaned_data = []
        for block in raw_data:
            # JSON format
            block = clean_json_format(block)
            items = re.findall(r"{.*?}", block, re.DOTALL)
            for item in items:
                try:
                    cleaned_data.append(json.loads(item))
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    print(f"Problematic block: {item}")
                    cleaned_data.append(item)
        return cleaned_data
      
    results = await asyncio.gather(*(_call_gpt(text, doc) for text, doc in zip(texts, docs)))
    return process(results)

text = "1960-1970年期間台灣的交通政策主要集中在以下幾個方面：交通建設的優先順序、民間投資的引入、交通基礎設施的發展、交通運輸的組織和管理"
#-----------------------pipeline

uf_list = gpt_unitfact_extraction(text)
queries = query_expansion(uf_list)
documents = retriever_api(queries)
verified_text = asyncio.run(gpt_txt_verification(uf_list, documents))
    
#-----------------------pipeline

print(verified_text) 
