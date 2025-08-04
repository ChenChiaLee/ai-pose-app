# 使用官方 Python 3.11 映像檔作為基礎
FROM python:3.11

# 設定工作目錄
WORKDIR /app

# 將您的 Streamlit 程式碼和 requirements.txt 複製到容器中
COPY . /app

# 安裝 apt 依賴 (您在 packages.txt 中指定的)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    mesa-vulkan-drivers

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 公開 Streamlit 應用程式的埠號
EXPOSE 8501

# 啟動 Streamlit 應用程式
CMD ["streamlit", "run", "streamlit_r1.py"]
