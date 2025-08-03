import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import mediapipe as mp
import joblib
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import tempfile
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shutil

# 設定頁面配置
st.set_page_config(
    page_title="AI 姿勢評估系統",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定義自訂損失函數
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# 使用 os.symlink 創建軟連結來解決權限問題
def setup_mediapipe_model():
    """
    此函數會透過建立軟連結，將 mediapipe 的模型路徑重新導向到一個可寫入的目錄，
    以解決 Streamlit Cloud 上的 Permission denied 錯誤。
    """
    try:
        # Streamlit Cloud 上的 mediapipe 預設模型目錄
        src_dir = '/home/adminuser/venv/lib/python3.11/site-packages/mediapipe/modules/pose_landmark/'
        
        # 我們可寫入的暫存目錄
        dest_dir = '/tmp/mediapipe_pose_models/'
        
        # 確保目標目錄存在
        os.makedirs(dest_dir, exist_ok=True)
        
        # 模型檔案在 GitHub 儲存庫的根目錄
        repo_model_path = os.path.join(os.getcwd(), 'pose_landmark_heavy.tflite')
        
        if not os.path.exists(repo_model_path):
            st.error("❌ 找不到專案根目錄中的 'pose_landmark_heavy.tflite' 檔案，請確保已上傳至 GitHub。")
            return False
            
        # 將模型檔案複製到我們的暫存目錄
        shutil.copyfile(repo_model_path, os.path.join(dest_dir, 'pose_landmark_heavy.tflite'))
        
        # 建立一個軟連結，將 mediapipe 導向到我們的暫存目錄
        # 注意：我們需要先檢查連結是否已存在，否則會報錯
        if not os.path.exists(src_dir):
            os.makedirs(src_dir, exist_ok=True)
            
        symlink_path = os.path.join(src_dir, 'pose_landmark_heavy.tflite')
        
        if not os.path.exists(symlink_path):
            os.symlink(os.path.join(dest_dir, 'pose_landmark_heavy.tflite'), symlink_path)
        
        st.sidebar.success("✅ mediapipe 模型路徑已成功修復！")
        return True
    
    except Exception as e:
        st.sidebar.error(f"❌ 修復 mediapipe 模型路徑時發生錯誤: {e}")
        return False


class PoseEvaluator:
    def __init__(self, model_path: str, scaler_path: str):
        # 載入模型時，將自訂函數傳入 custom_objects
        self.model = keras.models.load_model(model_path, custom_objects={'rmse': rmse})
        self.scaler = joblib.load(scaler_path)
        self.mp_pose = mp.solutions.pose
        
        # 載入 mediapipe 姿勢模型
        # model_complexity=2 對應 mediapipe 的 pose_landmark_heavy.tflite
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> List[float]:
        """處理單個影格並提取關鍵點（使用相對座標）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])

            # 獲取左髖 (23) 和右髖 (24) 的座標
            left_hip = landmarks[23, :3]
            right_hip = landmarks[24, :3]
            hip_center = (left_hip + right_hip) / 2

            # 獲取左肩 (11) 和右肩 (12) 的座標
            left_shoulder = landmarks[11, :3]
            right_shoulder = landmarks[12, :3]
            shoulder_center = (left_shoulder + right_shoulder) / 2

            # 計算標準化尺度
            scale = np.linalg.norm(shoulder_center - hip_center)
            if scale == 0:
                scale = 1

            # 將所有關鍵點轉換為相對於髖關節中心的座標，並進行尺度標準化
            normalized_landmarks = (landmarks[:, :3] - hip_center) / scale
            normalized_landmarks = np.hstack((normalized_landmarks, landmarks[:, 3:4]))

            return normalized_landmarks.flatten().tolist()
        return None

    def analyze_predictions(self, predictions: np.ndarray) -> dict:
        """分析預測結果並返回詳細統計資訊"""
        predictions = predictions.flatten()

        return {
            'min': np.min(predictions),
            'max': np.max(predictions),
            'mean': np.mean(predictions),
            'median': np.median(predictions),
            'std': np.std(predictions),
            '25th_percentile': np.percentile(predictions, 25),
            '75th_percentile': np.percentile(predictions, 75)
        }

    def evaluate_video(self, video_path: str, progress_callback=None) -> Tuple[float, dict, np.ndarray]:
        """評估影片中的動作並返回詳細分析"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        keypoints_list = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if progress_callback:
                progress_callback(frame_count / total_frames)

            keypoints = self.process_frame(frame)
            if keypoints:
                keypoints_list.append(keypoints)

        cap.release()

        if len(keypoints_list) == 0:
            return None, None, None

        # 數據預處理
        keypoints_array = np.array(keypoints_list)
        keypoints_array = self.scaler.transform(keypoints_array)
        keypoints_array = keypoints_array.reshape(-1, 33, 4)

        # 進行預測
        predictions = self.model.predict(keypoints_array, batch_size=32, verbose=0)
        stats = self.analyze_predictions(predictions)

        # 添加額外統計資訊
        stats.update({
            'total_frames': total_frames,
            'effective_frames': len(keypoints_list),
            'fps': fps,
            'duration': total_frames/fps,
            'detection_rate': (len(keypoints_list)/total_frames)*100
        })

        return stats['mean'], stats, predictions

def create_interactive_plots(predictions):
    """創建互動式圖表"""
    predictions_flat = predictions.flatten()

    # 創建子圖
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('預測值隨時間變化', '預測值分布'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. 預測值隨時間變化
    fig.add_trace(
        go.Scatter(y=predictions_flat, mode='lines', name='預測值', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_hline(y=np.mean(predictions_flat), line_dash="dash", line_color="red",
                  annotation_text="平均值", row=1, col=1)

    # 2. 預測值分布直方圖
    fig.add_trace(
        go.Histogram(x=predictions_flat, name='分布', nbinsx=30, opacity=0.7),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=True, title_text="姿勢評估分析結果")
    return fig

def main():
    st.title("🏃‍♂️ AI 姿勢評估系統")
    st.markdown("---")

    # 在載入模型前先進行 mediapipe 的路徑修復
    if not setup_mediapipe_model():
        st.sidebar.error("❌ 系統初始化失敗，無法修復 mediapipe 模型路徑。")
        st.stop()
        
    # 側邊欄 - 設定參數
    st.sidebar.header("⚙️ 系統設定")

    # 模型檔案路徑設定
    model_path = st.sidebar.text_input(
        "模型檔案路徑",
        value="CNN_squat_best.keras",
        help="請輸入訓練好的 Keras 模型檔案路徑"
    )

    scaler_path = st.sidebar.text_input(
        "標準化器檔案路徑",
        value="scaler_CNN_squat_best.pkl",
        help="請輸入用於資料標準化的 scaler 檔案路徑"
    )
    
    # 檢查檔案是否存在
    files_exist = all([
        os.path.exists(model_path) if model_path else False,
        os.path.exists(scaler_path) if scaler_path else False
    ])

    if not files_exist:
        st.error("❌ 請確認模型檔案和標準化器檔案路徑正確")
        st.stop()

    # 初始化評估器
    try:
        with st.spinner("正在載入模型..."):
            evaluator = PoseEvaluator(model_path, scaler_path)
        st.sidebar.success("✅ 模型載入成功")
    except Exception as e:
        st.sidebar.error(f"❌ 模型載入失敗: {str(e)}")
        st.stop()

    # 主要內容區域
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 影片上傳與分析")

        # 影片上傳
        uploaded_file = st.file_uploader(
            "選擇要分析的影片檔案",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="支援格式：MP4, AVI, MOV, MKV"
        )

        if uploaded_file is not None:
            # 儲存上傳的檔案到臨時位置
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name

            st.success(f"✅ 影片已上傳: {uploaded_file.name}")

            # 顯示影片資訊
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.info(f"📊 影片資訊: {duration:.2f}秒 | {total_frames}幀 | {fps:.1f} FPS | {width}x{height}")

            # 分析按鈕
            if st.button("🚀 開始分析", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"分析進度: {progress*100:.1f}%")

                try:
                    # 執行分析
                    avg_score, detailed_stats, predictions = evaluator.evaluate_video(
                        temp_video_path,
                        progress_callback=update_progress
                    )

                    if avg_score is not None:
                        status_text.text("✅ 分析完成！")

                        # 儲存結果到 session state
                        st.session_state['analysis_results'] = {
                            'avg_score': avg_score,
                            'detailed_stats': detailed_stats,
                            'predictions': predictions,
                            'video_name': uploaded_file.name
                        }
                    else:
                        st.error("❌ 未檢測到有效的姿勢數據，請檢查影片品質")

                except Exception as e:
                    st.error(f"❌ 分析過程中發生錯誤: {str(e)}")

                finally:
                    # 清理臨時檔案
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)

    with col2:
        st.header("📈 即時統計")

        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            stats = results['detailed_stats']

            # 顯示主要分數
            st.metric(
                label="平均姿勢分數",
                value=f"{results['avg_score']:.2f}",
                delta=f"±{stats['std']:.2f}"
            )

            # 顯示其他統計資訊
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("最高分", f"{stats['max']:.2f}")
                st.metric("檢測率", f"{stats['detection_rate']:.1f}%")
            with col2_2:
                st.metric("最低分", f"{stats['min']:.2f}")
                st.metric("影片長度", f"{stats['duration']:.1f}秒")

    # 結果視覺化區域
    if 'analysis_results' in st.session_state:
        st.markdown("---")
        st.header("📊 詳細分析結果")

        results = st.session_state['analysis_results']
        predictions = results['predictions']

        # 創建互動式圖表
        fig = create_interactive_plots(predictions)
        st.plotly_chart(fig, use_container_width=True)

        # 詳細統計表格
        st.subheader("📋 統計摘要")
        stats_df = pd.DataFrame([
            ['影片名稱', results['video_name']],
            ['平均分數', f"{results['avg_score']:.2f}"],
            ['標準差', f"{results['detailed_stats']['std']:.2f}"],
            ['最小值', f"{results['detailed_stats']['min']:.2f}"],
            ['最大值', f"{results['detailed_stats']['max']:.2f}"],
            ['中位數', f"{results['detailed_stats']['median']:.2f}"],
            ['25th 百分位', f"{results['detailed_stats']['25th_percentile']:.2f}"],
            ['75th 百分位', f"{results['detailed_stats']['75th_percentile']:.2f}"],
            ['總幀數', f"{results['detailed_stats']['total_frames']}"],
            ['有效幀數', f"{results['detailed_stats']['effective_frames']}"],
            ['檢測率', f"{results['detailed_stats']['detection_rate']:.1f}%"],
            ['影片長度', f"{results['detailed_stats']['duration']:.2f} 秒"]
        ], columns=['項目', '數值'])

        st.dataframe(stats_df, use_container_width=True)
            
if __name__ == "__main__":
    main()