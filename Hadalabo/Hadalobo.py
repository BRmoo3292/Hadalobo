from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Response, Request
import cv2
import numpy as np
import os
import tempfile
import asyncio
from openai import AsyncOpenAI
import aiohttp
import base64
from io import BytesIO
from PIL import Image
import time
import logging
import structlog

app = FastAPI(title="Hadalabo")

# 静的ファイル配信設定
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
if OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
else:
    logger.warning("OPENAI_API_KEY not found")
    openai_client = None

# 状態管理（物体検出は常時有効なので object_detection_active を削除）
conversation_active = False
conversation_task = None
latest_audio_filename = None
listening_for_speech = False
speech_recognized = False
recognized_text = ""

async def make_gpu_request(image_bytes):
    """GPUサーバーにリクエストを送信"""
    if not GPU_INFERENCE_URL:
        logger.warning("GPU_INFERENCE_URL not configured")
        return False
        
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
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"ngrok-skip-browser-warning": "true"}
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
    """リトライ機能付きGPU推論"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await make_gpu_request(image_bytes)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                logger.error(f"GPU inference failed after {max_retries} attempts: {e}")
                return False

@app.post("/predict")
async def predict(file: UploadFile):
    global conversation_active, conversation_task
    
    conversation_just_started = False
    img_bytes = await file.read()
    
    # 🎯 常に物体検出を実行
    objects_detected = await send_image_to_gpu_server(img_bytes)
    
    # 🛡️ 会話中でない場合のみ新しい会話を開始
    if objects_detected and not conversation_active:
        conversation_active = True
        conversation_just_started = True            
        logger.info("物体を検出。会話を開始します。")            
        conversation_task = asyncio.create_task(auto_converse())
    elif objects_detected and conversation_active:
        # 会話中に物体検出された場合はログのみ
        logger.info("会話中のため物体検出を無視しました")

    return JSONResponse(content={
        "conversation_started": conversation_just_started,
        "objects_detected": objects_detected,
        "conversation_active": conversation_active,
        "detection_ignored": objects_detected and conversation_active  # デバッグ用
    })

async def auto_converse():
    global conversation_active, latest_audio_filename

    fixed_messages = [
        "今日も一日おつかれさまでした。今はスキンケア中かな？肌にやさしく触れながら、今日のこと、少しお話ししませんか？",
        "まずは、今日一日を通して、いちばん心に残っていることは何ですか？",
        "今日、あなたができたなと思えることって、何かありましたか？",
        "心のエネルギーは何％くらい残っている感じですか？",
        "保肌も心も、今日はたくさん働いてくれました。保湿もできたし、あとはゆっくりおやすみタイムですね。",
        "よかったら、明日に向けたひとこと目標を決めてみませんか？明日は、どんな一日にしたいですか？",
        "では、今日はこのへんで。おやすみなさい。また明日、お話しできるのを楽しみにしています。"
    ]
    
    for index, reply_text in enumerate(fixed_messages):
        if not conversation_active:
            break
            
        logger.info(f"会話応答 {index+1}: {reply_text}")
        
        # TTS音声生成
        if openai_client:
            try:
                async with openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="nova",
                    input=reply_text,
                    response_format="mp3",
                    speed=1.0,
                ) as response:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=temp_dir) as tts_file:
                        async for chunk in response.iter_bytes():
                            tts_file.write(chunk)
                        latest_audio_filename = os.path.basename(tts_file.name)
                        logger.info(f"音声ファイル作成: {latest_audio_filename}")

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"TTS音声生成エラー: {e}")
                await asyncio.sleep(5)
        else:
            logger.warning("OpenAI client not available, skipping TTS")
            await asyncio.sleep(5)

    # 会話終了
    logger.info("会話終了")
    conversation_active = False

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
    global conversation_active, listening_for_speech
    conversation_active = False
    listening_for_speech = False
    logger.info("会話を停止しました。物体検出は継続します。")
    return {"message": "Conversation stopped, object detection continues"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "gpu_url_configured": bool(GPU_INFERENCE_URL),
        "openai_configured": bool(OPENAI_API_KEY),
        "conversation_active": conversation_active,
        "object_detection": "always_active"  # 常に有効
    }

@app.get("/status")
async def get_status():
    return {
        "conversation_active": conversation_active,
        "object_detection": "always_active",
        "timestamp": time.time()
    }

@app.get("/")
async def read_root():
    return FileResponse('Hadalobo.html', media_type='text/html')

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)