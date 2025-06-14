# 🚇 지하철 광고 예측 시스템

Prophet AI를 활용한 스마트 지하철 광고 전략 수립 웹 애플리케이션

## 📋 프로젝트 소개

이 프로젝트는 Streamlit과 Prophet을 활용하여 지하철역별 유동인구를 예측하고, 광고 효과를 분석하는 웹 애플리케이션입니다.

### 주요 기능

- 🤖 **AI 기반 예측**: Prophet 모델을 사용한 시계열 예측
- 📊 **인터랙티브 시각화**: Plotly를 활용한 동적 차트
- 🎯 **광고 효과 분석**: 5개 지하철역 동시 분석
- 📱 **현대적 UI**: 반응형 웹 인터페이스
- 📥 **결과 다운로드**: CSV 형태로 분석 결과 저장

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/YOUR_USERNAME/subway-ad-predictor.git
cd subway-ad-predictor
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 앱 실행
```bash
streamlit run predictAD.py
```

## 💻 사용법

1. **데이터 로딩**: 사이드바에서 "📊 데이터 로딩" 버튼 클릭
2. **역 선택**: 분석할 지하철역 5개 선택
3. **AI 분석**: "🤖 AI 분석 시작" 버튼으로 예측 실행
4. **결과 확인**: 다른 탭에서 분석 결과 및 시각화 확인

## 📊 분석 결과

- **5월 평균**: 실제 유동인구 데이터
- **6-7월 예측**: AI 예측 결과
- **변화율**: 증감률 분석
- **광고 판단**: 확대/유지/축소/중단 권장사항

## 🛠 기술 스택

- **Backend**: Python, Prophet
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data**: Pandas, NumPy

## 📁 프로젝트 구조

```
├── predictAD.py          # 메인 애플리케이션
├── requirements.txt      # 패키지 의존성
├── README.md            # 프로젝트 문서
└── *.csv               # 샘플 데이터 (선택적)
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 Issues를 통해 연락해주세요. 