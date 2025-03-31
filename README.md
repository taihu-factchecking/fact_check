# fact_check

## Fact-Check API Server
輸入台鵠系統的output("string")及檢索到的documents("https://nvcenter.ntu.edu.tw:8000/docs#/retrieve輸出的格式")，回傳是否被檢索到的文檔所支持。

### Request format example
```
url = "https://ntuairobo.net/taihu_factcheck"

params = {
  "text": "李侍堯對曾錦與王世昌的處理提出異議和建議",
  "docs": [
    {
      "page_content": "李侍堯在奏摺中指出......",
      "metadata": {
        "source": "52.txt",
        "chunk": 3,
        "start_idx": 0
      },
      "other": {}
    },
    # 接其他相關reference...
  ]
}
response = requests.post(url, json=params)
```

### Response format example
```
{
  "verification_results": [
    {
      "claim": "李侍堯對曾錦與王世昌的處理提出異議和建議",
      "factuality": true,
      "source": "52.txt",
      "idx": [0, 26],
      "docid": "",
      "url": "https://nvcenter.ntu.edu.tw:8000/full_text?title=52.txt"
    }
    # 更多結果會接續在後面
  ]
}
```
- claim: 查核的聲明，可能跟text中的句子不太一樣
- factuality: 若支持則為True，否則為False
- source: 支持的檔案名稱(在資料庫中的名稱)
- idx: claim在input的text中的index 
- docid: 省議會資料有自己的id，明清台灣行政檔案則無
- url: 可以看到檔案內容的網址



