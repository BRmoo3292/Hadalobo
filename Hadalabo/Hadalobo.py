from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Response, Request
import cv2
import numpy as np
import os
import tempfile
import asyncio
import google.generativeai as genai
from openai import AsyncOpenAI
import aiohttp
import base64
from io import BytesIO
from PIL import Image
import time
import requests
from dotenv import load_dotenv
import logging
import structlog

# 環境変数読み込み
load_dotenv()

app = FastAPI()

# 静的ファイル（HTMLや一時的な音声ファイル）を配信するための設定
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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# メトリクス収集
response_times = []
error_count = 0

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response_times.append(process_time)
    
    logger.info(
        "request_processed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        process_time=process_time
    )
    return response

# 環境変数から取得
GPU_INFERENCE_URL = os.environ.get("GPU_INFERENCE_URL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI クライアント初期化
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 会話状態を管理するフラグ
conversation_active = False
# 物体検出を停止するフラグ
object_detection_active = True

# 会話ループを管理するタスク
conversation_task = None

# 最新の音声ファイルパスを保存する変数
latest_audio_filename = None

# 音声認識の状態を管理
listening_for_speech = False
speech_recognized = False
recognized_text = ""

async def make_gpu_request(image_bytes):
    """GPUサーバーにリクエストを送信する関数"""
    try:
        # 画像をbase64エンコード
        img_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # PILに変換してbase64エンコード
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # GPUサーバーにリクエスト送信
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GPU_INFERENCE_URL,
                json={"image": img_base64},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("objects_detected", False)
                else:
                    logger.error(f"GPU server error: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"GPU server connection failed: {e}")
        return False

async def send_image_to_gpu_server(image_bytes):
    """画像をGPUサーバーに送信して物体検出を実行（リトライ機能付き）"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # GPU推論実行
            result = await make_gpu_request(image_bytes)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数バックオフ
                continue
            else:
                logger.error(f"GPU inference failed after {max_retries} attempts: {e}")
                return False

@app.post("/predict")
async def predict(file: UploadFile):
    global conversation_active, conversation_task, object_detection_active
    
    # 物体検出が無効になっている場合は処理をスキップ
    if not object_detection_active:
        return JSONResponse(content={
            "conversation_started": False,
            "object_detection_disabled": True
        })
    
    conversation_just_started = False
    img_bytes = await file.read()
    
    # GPU推論サーバーに画像を送信
    objects_detected = await send_image_to_gpu_server(img_bytes)
    
    if objects_detected:
        if not conversation_active:            
            conversation_active = True
            object_detection_active = False  # 物体検出を停止
            conversation_just_started = True            
            logger.info("物体を検出。会話を開始します。物体検出を停止します。")            
            conversation_task = asyncio.create_task(auto_converse())

    return JSONResponse(content={
        "conversation_started": conversation_just_started,
        "object_detection_disabled": not object_detection_active
    })

async def auto_converse():
    global conversation_active, latest_audio_filename, listening_for_speech, speech_recognized, recognized_text

    fixed_messages = [
        {"text": "今日も一日おつかれさまでした。今はスキンケア中かな？肌にやさしく触れながら、今日のこと、少しお話ししませんか？", "listen": False},
        {"text": "まずは、今日一日を通して、いちばん心に残っていることは何ですか？", "listen": True},
        {"text": "今日、あなたができたなと思えることって、何かありましたか？", "listen": True},
        {"text": "心のエネルギーは何％くらい残っている感じですか？", "listen": True},
        {"text": "保肌も心も、今日はたくさん働いてくれました。保湿もできたし、あとはゆっくりおやすみタイムですね。", "listen": False},
        {"text": "よかったら、明日に向けたひとこと目標を決めてみませんか？明日は、どんな一日にしたいですか？", "listen": True},
        {"text": "では、今日はこのへんで。おやすみなさい。また明日、お話しできるのを楽しみにしています。", "listen": False}
    ]
    index = 0
    
    while conversation_active:
        message_info = fixed_messages[index]
        reply_text = message_info["text"]
        should_listen = message_info["listen"]
        logger.info(f"会話応答: {reply_text}")
        
        # TTS音声生成
        try:
            async with openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=reply_text,
                response_format="mp3",
                speed=1.0,
            ) as response:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=temp_dir) as tts_file:
                    async for chunk in response.iter_bytes():
                        tts_file.write(chunk)
                    latest_audio_filename = os.path.basename(tts_file.name)
                    logger.info(f"音声ファイルを作成しました: {latest_audio_filename}")

                await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"TTS音声生成エラー: {e}")

        index += 1
        if index >= len(fixed_messages):
            logger.info("すべての定型文が終了しました。会話を終了します。")
            conversation_active = False
            break

@app.get("/get_audio")
async def get_audio():
    global latest_audio_filename
    if latest_audio_filename:
        audio_file_to_play = latest_audio_filename
        latest_audio_filename = None 
        return JSONResponse(content={"audio_url": f"/temp/{audio_file_to_play}"})
    return Response(status_code=204)

@app.post("/stop_conversation")
async def stop_conversation():
    global conversation_active, object_detection_active, listening_for_speech
    conversation_active = False
    object_detection_active = True
    listening_for_speech = False
    logger.info("会話を停止しました。物体検出を再開します。")
    return {"message": "Conversation stopped, object detection resumed"}

@app.get("/health")
async def health_check():
    """ヘルスチェック用エンドポイント"""
    return {"status": "healthy", "timestamp": time.time()}

def keep_session_alive():
    """定期的にダミーリクエストを送信（バックグラウンドタスク用）"""
    while True:
        try:
            # 軽量なヘルスチェック
            response = requests.get("http://localhost:8000/health")
            logger.info(f"Session alive: {response.status_code}")
        except Exception as e:
            logger.error(f"Session check failed: {e}")
        time.sleep(300)  # 5分間隔

@app.get("/")
def read_root():
    return FileResponse('Hadalobo.html', media_type='text/html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)