# 使用官方 Python 3.11 映像檔作為基礎
# Use the official Python 3.11 image as the base
FROM python:3.11

# 設定工作目錄
# Set the working directory
WORKDIR /app

# 將您的 Streamlit 程式碼和 requirements.txt 複製到容器中
# Copy your Streamlit code and requirements.txt into the container
COPY . /app

# 安裝 apt 依賴，包含 OpenCV 所需的函式庫
# Install apt dependencies, including libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    mesa-vulkan-drivers \
    libsm6 \
    libxext6 \
    ffmpeg

# 安裝 Python 依賴
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 公開 Streamlit 應用程式的埠號
# Expose the port for the Streamlit app
EXPOSE 8501

# 啟動 Streamlit 應用程式
# Start the Streamlit application
CMD ["streamlit", "run", "streamlit_r1.py"]
