#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚇 지하철 광고 예측 웹 앱
Streamlit 기반 현대적 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import time

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="🚇 지하철 광고 예측 시스템", 
    page_icon="🚇", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .station-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSubwayPredictor:
    def __init__(self):
        self.data = None
        self.models = {}
        self.predictions = {}
        self.ad_results = {}
        
    def install_requirements(self):
        """필요한 패키지 설치"""
        try:
            import prophet
            return True
        except ImportError:
            st.error("⚠️ Prophet 패키지가 설치되지 않았습니다.")
            st.code("pip install prophet")
            st.stop()
            return False
    
    def generate_sample_data(self):
        """샘플 데이터 생성"""
        stations = [
            "강남역", "홍대입구역", "신촌역", "이태원역", "명동역", 
            "종로3가역", "시청역", "을지로입구역", "동대문역사문화공원역", "신림역",
            "건대입구역", "성수역", "왕십리역", "청량리역", "잠실역",
            "신도림역", "사당역", "교대역", "역삼역", "선릉역",
            "압구정역", "논현역", "삼성역", "한양대역", "뚝섬역",
            "신설동역", "동대문역", "종각역", "안국역", "경복궁역",
            "독립문역", "서대문역", "충정로역", "아현역", "공덕역",
            "마포역", "합정역", "상수역", "광화문역", "을지로3가역",
            "충무로역", "동국대입구역", "약수역", "금고개역", "옥수역",
            "한남역", "노량진역", "대방역", "신대방역", "구로디지털단지역"
        ]
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 5, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_data = []
        
        for station in stations:
            station_seed = hash(station) % 10000
            np.random.seed(station_seed)
            
            base_popularity = np.random.choice([12000, 9000, 6000], p=[0.3, 0.5, 0.2])
            
            for i, date in enumerate(date_range):
                weekday_multiplier = {
                    0: 1.0, 1: 1.05, 2: 1.10, 3: 1.15, 4: 1.20, 5: 0.75, 6: 0.65
                }.get(date.weekday(), 1.0)
                
                month_multiplier = {
                    1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05
                }.get(date.month, 1.0)
                
                base_traffic = base_popularity * weekday_multiplier * month_multiplier
                trend = 30 * np.sin(2 * np.pi * i / 30)  # 트렌드 변동폭 줄임
                noise = np.random.normal(0, base_traffic * 0.02)  # 노이즈 크기 줄임 (0.05 -> 0.02)
                traffic = base_traffic + trend + noise
                
                min_traffic = base_traffic * 0.7
                max_traffic = base_traffic * 1.3
                traffic = max(min_traffic, min(max_traffic, traffic))
                
                all_data.append({
                    'ds': date,
                    'station': station,
                    'y': int(traffic)
                })
            
            np.random.seed()
        
        return pd.DataFrame(all_data)
    
    def clean_data(self, data):
        """데이터 정리 - 더 엄격한 이상치 제거 및 스무딩"""
        cleaned_data_list = []
        
        for station in data['station'].unique():
            station_data = data[data['station'] == station].copy()
            station_data = station_data.sort_values('ds').reset_index(drop=True)
            
            # 1단계: 더 엄격한 IQR 기반 이상치 제거
            Q1 = station_data['y'].quantile(0.25)
            Q3 = station_data['y'].quantile(0.75)
            IQR = Q3 - Q1
            
            # 이상치 범위를 더 좁게 설정 (1.5 -> 1.0)
            lower_bound = Q1 - 1.0 * IQR
            upper_bound = Q3 + 1.0 * IQR
            
            outlier_mask = (station_data['y'] >= lower_bound) & (station_data['y'] <= upper_bound)
            clean_data = station_data[outlier_mask].copy()
            
            # 2단계: 이동평균 스무딩 적용 (7일 윈도우)
            if len(clean_data) > 7:
                clean_data['y'] = clean_data['y'].rolling(window=7, center=True, min_periods=3).mean()
                clean_data = clean_data.dropna()
            
            # 3단계: Z-score 기반 추가 이상치 제거
            z_scores = np.abs((clean_data['y'] - clean_data['y'].mean()) / clean_data['y'].std())
            clean_data = clean_data[z_scores < 2.5]  # 2.5 표준편차 이내만 유지
            
            cleaned_data_list.append(clean_data)
        
        return pd.concat(cleaned_data_list, ignore_index=True)
    
    def train_models(self, selected_stations, progress_bar=None):
        """Prophet 모델 학습"""
        try:
            from prophet import Prophet
        except ImportError:
            st.error("Prophet 패키지를 설치해주세요: pip install prophet")
            return False
        
        for i, station in enumerate(selected_stations):
            if progress_bar:
                progress_bar.progress((i + 1) / len(selected_stations))
            
            station_data = self.data[self.data['station'] == station][['ds', 'y']].copy()
            station_data = station_data.sort_values('ds').reset_index(drop=True)
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                growth='linear',
                seasonality_mode='additive',
                changepoint_prior_scale=0.01,  # 더 부드러운 곡선을 위해 줄임 (0.05 -> 0.01)
                seasonality_prior_scale=3.0,   # 계절성 변동 줄임 (10.0 -> 3.0)
                holidays_prior_scale=3.0,      # 휴일 효과 줄임 (10.0 -> 3.0)
                mcmc_samples=0,
                uncertainty_samples=False
            )
            
            model.fit(station_data)
            self.models[station] = model
            
            # 예측 수행
            future = model.make_future_dataframe(periods=62)
            forecast = model.predict(future)
            self.predictions[station] = forecast
        
        return True
    
    def analyze_ad_effectiveness(self, selected_stations):
        """광고 효과 분석"""
        results = {}
        
        for station in selected_stations:
            forecast = self.predictions[station]
            
            # 5월 평균
            may_data = self.data[
                (self.data['station'] == station) & 
                (self.data['ds'] >= datetime(2025, 5, 1)) & 
                (self.data['ds'] < datetime(2025, 6, 1))
            ]['y'].mean()
            
            # 6-7월 예측 평균
            june_start = datetime(2025, 6, 1)
            future_data = forecast[forecast['ds'] >= june_start]
            future_predicted = future_data['yhat'].mean() if len(future_data) > 0 else may_data
            
            # 증감률 계산
            change_rate = ((future_predicted - may_data) / may_data) * 100 if may_data > 0 else 0
            
            # 판단 로직
            if change_rate >= 10:
                decision = "광고 확대 권장"
                color = "🟢"
                emoji = "📈"
            elif change_rate >= 5:
                decision = "광고 유지 권장"
                color = "🟢"
                emoji = "✅"
            elif change_rate >= -5:
                decision = "현상 유지"
                color = "🟡"
                emoji = "➡️"
            elif change_rate >= -10:
                decision = "광고 축소 고려"
                color = "🔴"
                emoji = "⚠️"
            else:
                decision = "광고 중단 권장"
                color = "🔴"
                emoji = "❌"
            
            results[station] = {
                'may_avg': may_data,
                'future_avg': future_predicted,
                'change_rate': change_rate,
                'decision': decision,
                'color': color,
                'emoji': emoji
            }
        
        self.ad_results = results
        return results

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚇 지하철 광고 예측 시스템</h1>
        <p>Prophet AI를 활용한 스마트 광고 전략 수립</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 예측 시스템 초기화
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitSubwayPredictor()
    
    predictor = st.session_state.predictor
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 데이터 로딩 버튼
        if st.button("📊 데이터 로딩", use_container_width=True):
            with st.spinner("데이터 로딩 중..."):
                predictor.data = predictor.generate_sample_data()
                predictor.data = predictor.clean_data(predictor.data)
                st.session_state.data_loaded = True
                st.success("✅ 데이터 로딩 완료!")
                st.rerun()
        
        st.divider()
        
        # Patent 설치 확인
        if predictor.install_requirements():
            st.success("✅ Prophet 패키지 준비됨")
        
        # 데이터 정보
        if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
            st.info(f"📅 데이터 기간: 2025.01.01 ~ 2025.05.31")
            st.info(f"🚇 총 {len(predictor.data['station'].unique())}개 역")
            st.info(f"📊 총 {len(predictor.data):,}개 데이터 포인트")
    
    # 메인 콘텐츠
    if not hasattr(st.session_state, 'data_loaded'):
        st.info("👈 먼저 사이드바에서 '데이터 로딩' 버튼을 클릭해주세요!")
        return
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🎯 역 선택 & 분석", "📊 상세 결과", "📈 시각화"])
    
    with tab1:
        st.header("🚇 분석할 지하철역 선택")
        
        # 역 선택
        available_stations = sorted(predictor.data['station'].unique())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_stations = st.multiselect(
                "5개 지하철역을 선택해주세요:",
                available_stations,
                placeholder="역을 선택하세요...",
                help="정확히 5개의 역을 선택해야 분석이 진행됩니다."
            )
        
        with col2:
            st.metric("선택된 역", f"{len(selected_stations)}/5")
            
            if len(selected_stations) == 5:
                st.success("✅ 선택 완료!")
            elif len(selected_stations) > 5:
                st.error("❌ 5개까지만 선택 가능!")
            else:
                st.info(f"ℹ️ {5-len(selected_stations)}개 더 선택하세요")
        
        # 선택된 역 표시
        if selected_stations:
            st.subheader("📍 선택된 역")
            cols = st.columns(min(len(selected_stations), 5))
            for i, station in enumerate(selected_stations):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div class="station-card">
                        <h4>🚇 {station}</h4>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 분석 실행
        if len(selected_stations) == 5:
            if st.button("🤖 AI 분석 시작", use_container_width=True, type="primary"):
                
                # 진행 상태 표시
                progress_container = st.container()
                with progress_container:
                    st.info("🚀 AI 분석을 시작합니다...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 모델 학습
                    status_text.text("🤖 Prophet 모델 학습 중...")
                    success = predictor.train_models(selected_stations, progress_bar)
                    
                    if success:
                        status_text.text("📊 광고 효과 분석 중...")
                        time.sleep(1)
                        predictor.analyze_ad_effectiveness(selected_stations)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 분석 완료!")
                        
                        st.session_state.analysis_complete = True
                        st.session_state.selected_stations = selected_stations
                        
                        # 성공 메시지
                        st.markdown("""
                        <div class="success-card">
                            <h3>🎉 분석 완료!</h3>
                            <p>다른 탭에서 상세 결과를 확인하세요.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(2)
                        st.rerun()
    
    with tab2:
        if not hasattr(st.session_state, 'analysis_complete'):
            st.info("먼저 '역 선택 & 분석' 탭에서 분석을 완료해주세요!")
            return
        
        st.header("📊 광고 효과 분석 결과")
        
        # 결과 카드 표시
        for i, station in enumerate(st.session_state.selected_stations):
            result = predictor.ad_results[station]
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            
            with col1:
                st.metric(
                    "5월 평균",
                    f"{int(result['may_avg']):,}명"
                )
            
            with col2:
                st.metric(
                    "6-7월 예측",
                    f"{int(result['future_avg']):,}명",
                    f"{result['change_rate']:+.1f}%"
                )
            
            with col3:
                st.metric(
                    "변화율",
                    f"{result['change_rate']:+.1f}%"
                )
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{result['emoji']} {station}</h3>
                    <h4>{result['color']} {result['decision']}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
    
    with tab3:
        if not hasattr(st.session_state, 'analysis_complete'):
            st.info("먼저 '역 선택 & 분석' 탭에서 분석을 완료해주세요!")
            return
        
        st.header("📈 시각화")
        
        # 차트 생성
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=[f"🚇 {station}" for station in st.session_state.selected_stations],
            vertical_spacing=0.15
        )
        
        for i, station in enumerate(st.session_state.selected_stations):
            row = i + 1
            col = 1
            
            forecast = predictor.predictions[station]
            
            # 실제 데이터와 예측 데이터 분리
            actual_data = forecast[forecast['ds'] < datetime(2025, 6, 1)]
            predicted_data = forecast[forecast['ds'] >= datetime(2025, 6, 1)]
            
            # 실제 데이터 (1-5월)
            fig.add_trace(
                go.Scatter(
                    x=actual_data['ds'],
                    y=actual_data['yhat'],
                    mode='lines+markers',
                    name=f'{station} (실제)',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=4),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # 예측 데이터 (6-7월)
            fig.add_trace(
                go.Scatter(
                    x=predicted_data['ds'],
                    y=predicted_data['yhat'],
                    mode='lines+markers',
                    name=f'{station} (예측)',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=4),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title="🚇 지하철역별 유동인구 추이 및 예측",
            height=1800,
            showlegend=True,
            legend=dict(x=0.85, y=1.0)
        )
        
        # y축 제목 설정
        fig.update_yaxes(title_text="유동인구 (명)")
        fig.update_xaxes(title_text="날짜")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 요약 테이블
        st.subheader("📋 결과 요약")
        
        summary_data = []
        for station in st.session_state.selected_stations:
            result = predictor.ad_results[station]
            summary_data.append({
                '역명': station,
                '5월평균': f"{int(result['may_avg']):,}명",
                '6-7월예측': f"{int(result['future_avg']):,}명",
                '변화율': f"{result['change_rate']:+.1f}%",
                '판단': f"{result['emoji']} {result['decision']}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # 다운로드 버튼
        csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 결과 다운로드 (CSV)",
            data=csv,
            file_name=f"지하철광고분석_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main() 