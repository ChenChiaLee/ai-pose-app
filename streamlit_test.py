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

# 設定頁面配置
st.set_page_config(
    page_title="AI 姿勢評估系統",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PoseEvaluator:
    def __init__(self, model_path: str, scaler_path: str, ground_truth_path: str = None):
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        
        # 載入真實標籤資料
        self.ground_truth_df = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            self.ground_truth_df = pd.read_csv(ground_truth_path)

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

    def get_ground_truth(self, video_filename: str) -> float:
        """根據影片名稱從資料集中獲取真實標籤值"""
        if self.ground_truth_df is None:
            return None
            
        video_id = os.path.splitext(video_filename)[0]
        
        if 'video_id' in self.ground_truth_df.columns:
            matching_row = self.ground_truth_df[self.ground_truth_df['video_id'] == video_id]
            if not matching_row.empty:
                label_col = 'total_score'
                if label_col in self.ground_truth_df.columns:
                    return matching_row[label_col].values[0]
                else:
                    numeric_cols = matching_row.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        return matching_row[numeric_cols[0]].values[0]
        return None

    def get_frame_ground_truths(self, video_id: str) -> List[float]:
        """獲取特定影片中每一幀的真實標籤分數"""
        if self.ground_truth_df is None:
            return []
    
        if {'video_id', 'frame', 'knee_score'}.issubset(self.ground_truth_df.columns):
            df = self.ground_truth_df
            frame_scores = df[df['video_id'] == video_id].sort_values(by='frame')['knee_score'].tolist()
            return frame_scores
        return []

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

def create_interactive_plots(predictions, ground_truth=None, frame_truths=None):
    """創建互動式圖表"""
    predictions_flat = predictions.flatten()
    
    # 創建子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('預測值隨時間變化', '預測值分布', '統計摘要'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "table"}, {"secondary_y": False}]]
    )
    
    # 1. 預測值隨時間變化
    fig.add_trace(
        go.Scatter(y=predictions_flat, mode='lines', name='預測值', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_hline(y=np.mean(predictions_flat), line_dash="dash", line_color="red", 
                  annotation_text="平均值", row=1, col=1)
    
    if ground_truth is not None:
        fig.add_hline(y=ground_truth, line_dash="solid", line_color="green", 
                      annotation_text="真實值", row=1, col=1)
    
    # 2. 預測值分布直方圖
    fig.add_trace(
        go.Histogram(x=predictions_flat, name='分布', nbinsx=30, opacity=0.7),
        row=1, col=2
    )
    
    # 3. 統計摘要表格
    stats_data = [
        ['平均值', f'{np.mean(predictions_flat):.2f}'],
        ['中位數', f'{np.median(predictions_flat):.2f}'],
        ['標準差', f'{np.std(predictions_flat):.2f}'],
        ['最小值', f'{np.min(predictions_flat):.2f}'],
        ['最大值', f'{np.max(predictions_flat):.2f}']
    ]
    
    if ground_truth is not None:
        stats_data.append(['真實值', f'{ground_truth:.2f}'])
        stats_data.append(['誤差', f'{abs(np.mean(predictions_flat) - ground_truth):.2f}'])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['統計項目', '數值']),
            cells=dict(values=[[row[0] for row in stats_data], 
                              [row[1] for row in stats_data]])
        ),
        row=2, col=1
    )
    '''
    # 4. 如果有逐幀真實值，顯示比較
    if frame_truths and len(frame_truths) == len(predictions_flat):
        fig.add_trace(
            go.Scatter(y=predictions_flat, mode='lines', name='預測值', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=frame_truths, mode='lines', name='真實值', line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # 計算相關係數
        corr, _ = pearsonr(predictions_flat, frame_truths)
        fig.add_annotation(
            text=f"相關係數: {corr:.3f}",
            xref="x4", yref="y4",
            x=len(predictions_flat)*0.7, y=max(max(predictions_flat), max(frame_truths))*0.9,
            showarrow=False,
            bgcolor="white",
            bordercolor="black"
        )
    '''
    fig.update_layout(height=800, showlegend=True, title_text="姿勢評估分析結果")
    return fig

def main():
    st.title("🏃‍♂️ AI 姿勢評估系統")
    st.markdown("---")
    
    # 側邊欄 - 設定參數
    st.sidebar.header("⚙️ 系統設定")
    
    # 模型檔案路徑設定
    model_path = st.sidebar.text_input(
        "模型檔案路徑", 
        value="Alexnet_squat0603.keras",
        help="請輸入訓練好的 Keras 模型檔案路徑"
        )

    scaler_path = st.sidebar.text_input(
        "標準化器檔案路徑", 
        value="scaler_Alexnet_squat0603.pkl",
        help="請輸入用於資料標準化的 scaler 檔案路徑"
        )
    '''
    ground_truth_path = st.sidebar.text_input(
        "真實標籤檔案路徑（可選）", 
        value="squat_400(0603).csv",
        help="如果有真實標籤資料，請輸入 CSV 檔案路徑"
        )
    '''
    
    # 新增以下這行，用來顯示Streamlit Cloud上的檔案列表
    st.sidebar.text(f"當前目錄中的檔案:\n{os.listdir('.')}")
    
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
                        
                        # 獲取真實標籤（如果有）
                        ground_truth = evaluator.get_ground_truth(uploaded_file.name)
                        video_id = os.path.splitext(uploaded_file.name)[0]
                        frame_truths = evaluator.get_frame_ground_truths(video_id)
                        
                        st.session_state['ground_truth'] = ground_truth
                        st.session_state['frame_truths'] = frame_truths
                        
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
            
            # 真實值比較（如果有）
            if 'ground_truth' in st.session_state and st.session_state['ground_truth'] is not None:
                ground_truth = st.session_state['ground_truth']
                error = abs(results['avg_score'] - ground_truth)
                relative_error = (error / ground_truth) * 100
                
                st.markdown("### 🎯 準確度分析")
                st.metric("真實分數", f"{ground_truth:.2f}")
                st.metric("絕對誤差", f"{error:.2f}")
                st.metric("相對誤差", f"{relative_error:.1f}%")
    
    # 結果視覺化區域
    if 'analysis_results' in st.session_state:
        st.markdown("---")
        st.header("📊 詳細分析結果")
        
        results = st.session_state['analysis_results']
        predictions = results['predictions']
        ground_truth = st.session_state.get('ground_truth')
        frame_truths = st.session_state.get('frame_truths')
        
        # 創建互動式圖表
        fig = create_interactive_plots(predictions, ground_truth, frame_truths)
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
'''       
        # 下載結果
        if st.button("📥 下載分析結果"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 準備下載數據
            download_data = {
                'video_name': results['video_name'],
                'analysis_time': timestamp,
                'average_score': results['avg_score'],
                'detailed_stats': results['detailed_stats'],
                'frame_predictions': predictions.flatten().tolist()
            }
            
            if ground_truth is not None:
                download_data['ground_truth'] = ground_truth
                download_data['absolute_error'] = abs(results['avg_score'] - ground_truth)
            
            # 轉換為 JSON 字符串
            import json
            json_str = json.dumps(download_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="下載 JSON 格式結果",
                data=json_str,
                file_name=f"pose_analysis_{timestamp}.json",
                mime="application/json"
            )
'''
if __name__ == "__main__":
    main()