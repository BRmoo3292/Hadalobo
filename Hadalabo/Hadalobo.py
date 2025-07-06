from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile,Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import asyncio
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
model = YOLO('best.pt')  # YOLOモデルのパスを変更してください
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-2.0-flash")    
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(frame, imgsz=320, conf=0.75)
    if len(results) > 0 and len(results[0].boxes) > 0:
        if not conversation_active:            
            conversation_active = True
            object_detection_active = False  # 物体検出を停止
            conversation_just_started = True            
            print("[INFO] 物体を検出。会話を開始します。物体検出を停止します。")            
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
    index = 0  # 現在のメッセージインデックス
    
    while conversation_active:
        message_info = fixed_messages[index]
        reply_text = message_info["text"]
        should_listen = message_info["listen"]
        print(f"[会話] 応答: {reply_text}")
        
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
                latest_audio_filename = os.path.basename(tts_file.name)
                print(f"[INFO] 音声ファイルを作成しました: {latest_audio_filename}")

            await asyncio.sleep(10)


        index += 1
        if index >= len(fixed_messages):
                print("[INFO] すべての定型文が終了しました。会話を終了します。")
                conversation_active = False
                break

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
    global conversation_active, object_detection_active, listening_for_speech
    conversation_active = False
    object_detection_active = True  # 物体検出を再開
    listening_for_speech = False
    print("[INFO] 会話を停止しました。物体検出を再開します。")
    return {"message": "Conversation stopped, object detection resumed"}

@app.get("/")
def read_root():
    return FileResponse('Hadalobo.html', media_type='text/html')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)