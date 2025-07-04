from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile,Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import tempfile
import asyncio
import time
import google.generativeai as genai
from openai import AsyncOpenAI

app = FastAPI()

# 静的ファイル（HTMLや一時的な音声ファイル）を配信するための設定
# この設定により、一時的に作成した音声ファイルをブラウザからアクセスできるようになります。
temp_dir = tempfile.gettempdir()
app.mount("/temp", StaticFiles(directory=temp_dir), name="temp")


# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルロード
model =  YOLO('best.pt')  # YOLOモデルのパスを変更してください
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-2.0-flash")
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 会話状態を管理するフラグ
conversation_active = False

# 会話ループを管理するタスク
conversation_task = None

# 最新の音声ファイルパスを保存する変数
latest_audio_filename = None

@app.post("/predict")
async def predict(file: UploadFile):
    global conversation_active, conversation_task
    
    conversation_just_started = False

    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(frame, imgsz=(320, 320))


    if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    if not conversation_active:
                        conversation_active = True
                        conversation_just_started = True
                        print("[INFO] 物体を検出。会話を開始します。")
                        conversation_task = asyncio.create_task(auto_converse())

    return JSONResponse(content={
        "conversation_started": conversation_just_started,
    })

async def auto_converse():
    global conversation_active, latest_audio_filename

    while conversation_active:
        prompt = """
あなたは、ユーザーの毎日のスキンケアを応援する、賢くて優しい「化粧水のボトル」です。
これからスキンケアやメイクを始めるユーザーに向けて、アドバイスや応援の言葉をかけてあげてください。
# あなたの役割
- スキンケアの専門家として、ユーザーに寄り添うパートナーです。
- ユーザーがもっと自分の肌を好きになれるように、ポジティブな気持ちにさせることが目的です。

# 話し方のルール
- 口調は、親しみやすく、丁寧な「ですます調」を基本とします。
- 1回の文字程度の短くて分かりやすい一言にしてください。
- ユーザーを励ましたり、褒めたり、具体的なアドバイスをしたりします。

"""
        response = model_gemini.generate_content(prompt)
        reply_text = response.text.strip()

        print("[会話] 応答:", reply_text)

        # TTS音声生成
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=reply_text,
            response_format="mp3",
            speed=1.0,
        ) as response:
            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=temp_dir) as tts_file:
                async for chunk in response.iter_bytes():
                    tts_file.write(chunk)
                # ファイル名を保存
                latest_audio_filename = os.path.basename(tts_file.name)
                print(f"[INFO] 音声ファイルを作成しました: {latest_audio_filename}")

        await asyncio.sleep(10)  # 次の発話までの待機時間（10秒）

@app.get("/get_audio")
async def get_audio():
    global latest_audio_filename
    if latest_audio_filename:
        audio_file_to_play = latest_audio_filename
        latest_audio_filename = None 
        return JSONResponse(content={"audio_url": f"/temp/{audio_file_to_play}"})
    # 新しい音声がない場合は、ボディが空の204レスポンスを返す
    return Response(status_code=204)

@app.post("/stop_conversation")
async def stop_conversation():
    global conversation_active
    conversation_active = False
    print("[INFO] 会話を停止しました。")
    return {"message": "Conversation stopped"}

@app.get("/")
def read_root():
    return FileResponse('Hadalobo.html', media_type='text/html')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
