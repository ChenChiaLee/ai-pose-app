import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import joblib
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import tempfile
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 設定頁面配置
st.set_page_config(
    page_title="AI 姿勢評估系統",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定義自訂的 RMSE 損失函數
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# 定義檔案路徑，假設檔案已存在於專案根目錄
MODEL_PATH = "CNN_squat_best.keras"
SCALER_PATH = "scaler_CNN_squat_best.pkl"
POSE_MODEL_PATH = "pose_landmark_heavy.tflite"

class PoseEvaluator:
    def __init__(self, model_path: str, scaler_path: str):
        # 使用 custom_objects 參數載入模型
        self.model = keras.models.load_model(model_path, custom_objects={'rmse': rmse})
        self.scaler = joblib.load(scaler_path)

        # 設定 mediapipe 使用 CPU
        base_options = mp_tasks.BaseOptions(
            model_asset_path=POSE_MODEL_PATH,
            delegate=mp_tasks.BaseOptions.Delegate.CPU
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> List[float]:
        """處理單個影格並提取關鍵點（使用相對座標）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_tasks.core.Image.create_from_numpy_array(rgb_frame)
        detection_result = self.pose_landmarker.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            landmarks_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            
            left_hip = landmarks_np[23, :3]
            right_hip = landmarks_np[24, :3]
            hip_center = (left_hip + right_hip) / 2
            
            left_shoulder = landmarks_np[11, :3]
            right_shoulder = landmarks_np[12, :3]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            scale = np.linalg.norm(shoulder_center - hip_center)
            if scale == 0:
                scale = 1
                
            normalized_landmarks = (landmarks_np[:, :3] - hip_center) / scale
            normalized_landmarks = np.hstack((normalized_landmarks, landmarks_np[:, 3:4]))
            
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
            
        keypoints_array = np.array(keypoints_list)
        keypoints_array = self.scaler.transform(keypoints_array)
        keypoints_array = keypoints_array.reshape(-1, 33, 4)
        
        predictions = self.model.predict(keypoints_array, batch_size=32, verbose=0)
        stats = self.analyze_predictions(predictions)
        
        stats.update({
            'total_frames': total_frames,
            'effective_frames': len(keypoints_list),
            'fps': fps,
            'duration': total_frames/fps,
            'detection_rate': (len(keypoints_list)/total_frames)*100
        })
        
        return stats['mean'], stats, predictions

def create_interactive_plots(predictions):
    predictions_flat = predictions.flatten()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('預測值隨時間變化', '預測值分布'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    fig.add_trace(go.Scatter(y=predictions_flat, mode='lines', name='預測值', line=dict(color='blue')), row=1, col=1)
    fig.add_hline(y=np.mean(predictions_flat), line_dash="dash", line_color="red", annotation_text="平均值", row=1, col=1)
    fig.add_trace(go.Histogram(x=predictions_flat, name='分布', nbinsx=30, opacity=0.7), row=1, col=2)
    fig.update_layout(height=400, showlegend=True, title_text="姿勢評估分析結果")
    return fig

def main():
    st.title("🏃‍♂️ AI 姿勢評估系統")
    st.markdown("---")
    
    # 側邊欄 - 設定參數
    st.sidebar.header("⚙️ 系統設定")
    model_path = st.sidebar.text_input("模型檔案路徑", value=MODEL_PATH, help="訓練好的 Keras 模型檔案")
    scaler_path = st.sidebar.text_input("標準化器檔案路徑", value=SCALER_PATH, help="用於資料標準化的 scaler 檔案")
    
    files_exist = all([os.path.exists(model_path), os.path.exists(scaler_path), os.path.exists(POSE_MODEL_PATH)])
    
    if not files_exist:
        st.error("❌ 請確認模型檔案和標準化器檔案已存在")
        st.stop()
    
    try:
        with st.spinner("正在載入模型..."):
            evaluator = PoseEvaluator(model_path, scaler_path)
        st.sidebar.success("✅ 模型載入成功")
    except Exception as e:
        st.sidebar.error(f"❌ 模型載入失敗: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("📹 影片上傳與分析")
        uploaded_file = st.file_uploader(
            "選擇要分析的影片檔案",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="支援格式：MP4, AVI, MOV, MKV"
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            st.success(f"✅ 影片已上傳: {uploaded_file.name}")
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            st.info(f"📊 影片資訊: {duration:.2f}秒 | {total_frames}幀 | {fps:.1f} FPS | {width}x{height}")
            if st.button("🚀 開始分析", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"分析進度: {progress*100:.1f}%")
                try:
                    avg_score, detailed_stats, predictions = evaluator.evaluate_video(
                        temp_video_path,
                        progress_callback=update_progress
                    )
                    if avg_score is not None:
                        status_text.text("✅ 分析完成！")
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
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
    with col2:
        st.header("📈 即時統計")
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            stats = results['detailed_stats']
            st.metric(label="平均姿勢分數", value=f"{results['avg_score']:.2f}", delta=f"±{stats['std']:.2f}")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("最高分", f"{stats['max']:.2f}")
                st.metric("檢測率", f"{stats['detection_rate']:.1f}%")
            with col2_2:
                st.metric("最低分", f"{stats['min']:.2f}")
                st.metric("影片長度", f"{stats['duration']:.1f}秒")
    if 'analysis_results' in st.session_state:
        st.markdown("---")
        st.header("📊 詳細分析結果")
        results = st.session_state['analysis_results']
        predictions = results['predictions']
        fig = create_interactive_plots(predictions)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📋 統計摘要")
        stats_df = pd.DataFrame([
            ['影片名稱', results['video_name']], ['平均分數', f"{results['avg_score']:.2f}"],
            ['標準差', f"{results['detailed_stats']['std']:.2f}"], ['最小值', f"{results['detailed_stats']['min']:.2f}"],
            ['最大值', f"{results['detailed_stats']['max']:.2f}"], ['中位數', f"{results['detailed_stats']['median']:.2f}"],
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