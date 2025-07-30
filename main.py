from fastapi import FastAPI
from fastapi import UploadFile, File
import pytesseract
from PIL import Image
import io
import os
from fastapi import Body
import openai
from pdf2image import convert_from_bytes
import base64
import subprocess
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import re
import requests
from gtts import gTTS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174", 
        "https://*.vercel.app",
        "https://*.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "News Generator API is running."}

@app.post("/ocr")
def ocr_images(files: list[UploadFile] = File(...)):
    if len(files) > 7:
        return {"error": "最大7枚までアップロード可能です"}
    
    all_texts = []
    for file in files:
        filename = file.filename.lower()
        image_bytes = file.file.read()
        text = ""
        if filename.endswith(".pdf"):
            # PDFの場合、全ページ画像化してOCR
            images = convert_from_bytes(image_bytes)
            texts = []
            for img in images:
                t = pytesseract.image_to_string(img, lang="jpn+eng")
                texts.append(t)
            text = "\n".join(texts)
        else:
            # 画像ファイルの場合
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang="jpn+eng")
        all_texts.append(text)
    
    combined_text = "\n\n".join(all_texts)
    return {"text": combined_text}

@app.post("/generate_news")
def generate_news(
    title: str = Body(...),
    content: str = Body(...),
    additional_info: str = Body(""),
    duration: int = Body(30),
    news_type: str = Body("イベント紹介")
):
    # インタビュー指示の有無を判定
    interview_instruction = ""
    if "インタビュー" in additional_info or "interview" in additional_info.lower():
        interview_instruction = "（※インタビュー部分は記者の質問は含めず、関係者の回答のみを記載してください）"
    
    # ニュース種別に応じた指示を追加
    style_instruction = ""
    if news_type == "ストレートニュース":
        style_instruction = "（※事実を正確に伝える硬いニュース調で、客観的な表現を使用してください。取材は完了しており、放送用の原稿として作成してください。イベントや出来事は既に終了・完了したものとして過去形で表現してください）"
    elif news_type == "情報番組用":
        style_instruction = "（※視聴者に親しみやすく、分かりやすい表現で、やわらかい語りかけ調で作成してください。取材は完了しており、放送用の原稿として作成してください。イベントや出来事は既に終了・完了したものとして過去形で表現してください）"
    
    prompt = f"""
あなたはプロの報道記者です。以下の情報をもとに、{duration}秒で読める{news_type}のニュース原稿を作成してください。
{style_instruction}

【重要】イベントや出来事は既に終了・完了したものとして、過去形で表現してください。
「開催されました」「決定しました」「行われました」「終了しました」などの過去形・完了形を使用してください。
「開催されます」「決定します」「行われます」などの未来形・予定形は使用しないでください。

【原稿の長さについて】
指定された{duration}秒で読める長さに厳密に合わせてください。
- 20秒: 約100文字
- 30秒: 約150文字
- 1分（60秒）: 約300文字
- 1分30秒（90秒）: 約450文字
- 2分（120秒）: 約600文字
- 2分30秒（150秒）: 約750文字
- 3分（180秒）: 約900文字
- 4分（240秒）: 約1200文字

【タイトル】
{title}

【本文】
{content}

【追加情報】
{additional_info}
{interview_instruction}

【フォーマット例】
【見出し】
【リード文】
【詳細説明】
【関係者コメント】
【締めの言葉】
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7
        )
        news_text = response.choices[0].message.content.strip()
        return {"news": news_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/tts_text_to_speech")
async def tts_text_to_speech(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            return {"error": "textフィールドが空です"}
        
        # テキスト前処理: タイトル行（1行目）と【】部分を除去、空白正規化
        lines = text.splitlines()
        if len(lines) > 1:
            text_body = '\n'.join(lines[1:])
        else:
            text_body = text
        # 【】で囲まれた部分を除去
        text_body = re.sub(r'【.*?】', '', text_body)
        # 連続空白・改行を1つのスペースに
        text_body = re.sub(r'[\s\u3000]+', ' ', text_body).strip()
        
        # gTTSで音声生成（1.25倍速で自然な音声）
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            tts = gTTS(text=text_body, lang='ja', slow=False)
            tts.save(mp3_file.name)
            
            with open(mp3_file.name, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        return {"audio_base64": audio_base64}
    except Exception as e:
        return {"error": str(e)}

@app.post("/tts_voicevox")
async def tts_voicevox(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        speaker = int(data.get("speaker", 1))
        rate = float(data.get("rate", 1.0))
        pitch = float(data.get("pitch", 0.0))
        if not text:
            return {"error": "textフィールドが空です"}
        # テキスト前処理: タイトル行（1行目）と【】部分を除去、空白正規化
        lines = text.splitlines()
        if len(lines) > 1:
            text_body = '\n'.join(lines[1:])
        else:
            text_body = text
        # 【】で囲まれた部分を除去
        text_body = re.sub(r'【.*?】', '', text_body)
        # 連続空白・改行を1つのスペースに
        text_body = re.sub(r'[\s\u3000]+', ' ', text_body).strip()
        # VOICEVOX API: audio_query
        query_res = requests.post(
            "http://localhost:50021/audio_query",
            params={"text": text_body, "speaker": speaker},
        )
        if query_res.status_code != 200:
            return {"error": f"audio_queryエラー: {query_res.text}"}
        query = query_res.json()
        # パラメータ反映
        query["speedScale"] = rate
        query["pitchScale"] = pitch
        # VOICEVOX API: synthesis
        synth_res = requests.post(
            "http://localhost:50021/synthesis",
            params={"speaker": speaker},
            json=query
        )
        if synth_res.status_code != 200:
            return {"error": f"synthesisエラー: {synth_res.text}"}
        audio_base64 = base64.b64encode(synth_res.content).decode("utf-8")
        return {"audio_base64": audio_base64}
    except Exception as e:
        return {"error": str(e)} 