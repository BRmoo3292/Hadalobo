# 1. ベースイメージとして軽量なPythonを選択
FROM python:3.9-slim

# 2. 環境変数を設定
ENV PYTHONUNBUFFERED=1

# 3. 作業ディレクトリを作成し、移動
WORKDIR /app

# 4. requirements.txtをコピー
COPY requirements.txt .

# 5. CPU版のPyTorchと関連ライブラリをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. アプリケーションのソースコードをコピー
COPY . .

# 7. アプリケーションを実行
CMD ["uvicorn", "Hadalobo:app", "--host", "0.0.0.0", "--port", "8000"]
