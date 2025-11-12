# AI 기반 트레이딩 시스템 / AI-Based Trading System

[한국어](#한국어) | [English](#english)

---

## 한국어

## 프로젝트 개요

시장 데이터를 기반으로 한 AI 트레이딩 시스템 **샘플 프로젝트**입니다. XGBoost, LSTM, 강화학습을 활용한 역할 분담 앙상블 방식으로 거래 신호를 생성합니다.

**참고**: 현재 샘플 프로젝트에는 AI Learning Module과 백테스트 모듈만 구현되어 있으며, 실제 거래 실행 모듈과 Data Feedback/Optimizer Module은 향후 구현 예정입니다.

### 프로젝트 목표

- AI 트레이딩 시그널 생성 시스템의 샘플 구현 제공
- 역할 분담 앙상블 구조의 개념 및 구현 방법 시연
- 백테스트를 통한 모델 성능 평가

### 핵심 설계 원칙: 역할 분담 방식 (Role-Based Ensemble)

각 AI 모델이 명확한 역할을 담당하여 협력하는 구조:

- **XGBoost**: 방향 예측 (상승/하락 확률)
- **LSTM**: 가격 목표 예측 (목표가, 손절가)
- **강화학습**: 포지션 크기 및 타이밍 결정

## 전체 시스템 구성 (샘플 프로젝트)

```
┌─────────────────────────────────────────────────────────┐
│  데이터 수집 및 피처 엔지니어링                            │
│  - 시장 데이터 수집 (Yahoo Finance)                        │
│  - 기술적 지표 생성 (MA, MACD, RSI, Bollinger Bands 등)   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  AI Learning Module                                     │
│  - XGBoost: 방향 예측                                    │
│  - LSTM: 가격 예측                                       │
│  - 강화학습: 최종 행동 결정                                │
│  - 역할 분담 앙상블                                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  백테스트 모듈                                           │
│  - 과거 데이터로 거래 시뮬레이션                           │
│  - 성과 지표 계산 (수익률, 승률, Sharpe Ratio 등)         │
└─────────────────────────────────────────────────────────┘

참고: 실제 거래 실행 모듈과 Data Feedback/Optimizer Module은 향후 구현 예정입니다.
```

## 프로젝트 구조

```
role-based-trading-ai/
├── README.md                          # 프로젝트 개요 (본 문서)
├── LICENSE                            # MIT 라이선스
├── requirements.txt                   # Python 패키지 의존성
├── .gitignore                         # Git 무시 파일
├── docs/                              # 상세 기술 문서
│   ├── architecture.md               # AI 모델 아키텍처
│   ├── system_design.md              # 시스템 설계 및 인터페이스
│   └── implementation_guide.md       # 구현 가이드
└── example/                           # 샘플 프로젝트
    ├── run_sample.py                  # 샘플 실행 스크립트
    ├── config/                        # 설정 파일
    │   └── sample_config.yaml
    ├── data/                          # 데이터 저장소
    │   ├── download_data.py          # 데이터 다운로드 스크립트
    │   └── nvda_data.csv             # 샘플 데이터 (NVDA)
    ├── results/                       # 결과 저장소 (이미지, 차트 등)
    ├── feature_engineering/           # 피처 엔지니어링 모듈
    │   └── feature_engineering.py    # 기술적 지표 생성
    └── ai_learning/                   # AI 학습 모듈 샘플
        ├── models/                    # 모델 구현
        │   ├── xgboost_direction.py
        │   ├── lstm_price.py
        │   └── rl_agent.py
        ├── ensemble/                  # 앙상블
        │   └── role_based_ensemble.py
        ├── backtest.py                # 백테스트 모듈
        └── train_sample.py            # 샘플 학습 스크립트
```

## 빠른 시작

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장, 없어도 CPU로 동작)

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd role-based-trading-ai

# 가상환경 생성 및 활성화
python -m venv venv

# 가상환경 활성화
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 샘플 프로젝트 실행

```bash
cd example
python run_sample.py
```

샘플 프로젝트는 전체 파이프라인을 간단히 보여줍니다. NVDA 종목 데이터를 사용하여 XGBoost, LSTM, 강화학습 모델을 학습하고 백테스트를 수행합니다.

## 주요 기능

### AI Learning Module

1. **XGBoost 방향 예측 모델**: 구조화된 피처 기반 방향 예측
2. **LSTM 가격 예측 모델**: 시계열 데이터 기반 가격 목표 예측
3. **강화학습 Agent**: 최종 행동 및 포지션 크기 결정
4. **역할 분담 앙상블**: 각 모델의 예측을 통합하여 최종 신호 생성


## 문서

### 핵심 문서

- **[AI 모델 아키텍처](./docs/architecture.md)**: AI 모델 구조 및 역할 분담 앙상블 상세 설명
- **[시스템 설계](./docs/system_design.md)**: 시스템 구성, 인터페이스, 데이터 흐름, 컴포넌트 설계

### 상세 가이드

- **[구현 가이드](./docs/implementation_guide.md)**: 기술 스택, 단계별 구현 방법, GPU/CPU 자동 감지

## 기술 스택 요약

- **AI/ML**: XGBoost, PyTorch, Stable-Baselines3, Gymnasium
- **데이터 처리**: Pandas, NumPy, Scikit-learn
- **데이터 수집**: yfinance (Yahoo Finance)
- **시각화**: Matplotlib
- **언어**: Python 3.8+

**참고**: 샘플 프로젝트에는 MLflow, Prometheus, Grafana가 포함되어 있지 않습니다. 이는 향후 확장 시 사용할 수 있는 기술 스택입니다.

자세한 기술 스택은 [구현 가이드](./docs/implementation_guide.md)를 참고하세요.

## 중요 고지사항 (Disclaimer)

**본 프로젝트는 교육 및 연구 목적으로 제공됩니다.**

- 이 프로젝트는 **투자 조언이 아닙니다**. 실제 자금으로 거래하기 전에 충분한 검증과 테스트를 수행하세요.
- 과거 성과가 미래 수익을 보장하지 않습니다. 모든 투자에는 손실 위험이 따릅니다.
- 백테스트 결과는 실제 거래 환경과 다를 수 있습니다 (슬리피지, 거래 비용, 유동성 등).
- 본 소프트웨어를 사용하여 발생하는 모든 손실에 대해 프로젝트 작성자 및 기여자는 책임을 지지 않습니다.
- 실제 거래에 사용하기 전에 반드시 전문가의 조언을 구하시기 바랍니다.

## 라이선스

본 프로젝트는 [MIT License](./LICENSE) 하에 배포됩니다.

## 기여하기 (Contributing)

기여를 환영합니다! 다음 사항을 참고해주세요:

1. **이슈 리포트**: 버그나 개선 사항을 발견하셨다면 이슈를 등록해주세요.
2. **Pull Request**: 
   - 코드 스타일을 일관되게 유지해주세요.
   - 새로운 기능 추가 시 테스트와 문서화를 포함해주세요.
   - 커밋 메시지는 명확하게 작성해주세요.
3. **코드 리뷰**: 모든 PR은 코드 리뷰를 거칩니다.
4. **문서화**: 코드 변경 시 관련 문서도 함께 업데이트해주세요.

### 개발 가이드라인

- Python 코드 스타일: PEP 8 준수
- 타입 힌트 사용 권장
- Docstring 작성 (Google 스타일 권장)
- 단위 테스트 작성 권장

## 문의 및 지원

- 이슈 트래커를 통해 문의해주세요.
- 버그 리포트나 기능 제안을 환영합니다.

## 감사의 말

이 프로젝트에 기여해주신 모든 분들께 감사드립니다.

---

**면책 조항**: 본 소프트웨어는 "있는 그대로" 제공되며, 명시적이거나 묵시적인 어떠한 보증도 없습니다. 본 소프트웨어의 사용으로 인해 발생하는 모든 손실이나 손해에 대해 작성자나 기여자는 책임을 지지 않습니다.

---

## English

## Project Overview

An AI-based trading system **sample project** that generates trading signals using a role-based ensemble approach with XGBoost, LSTM, and Reinforcement Learning models based on market data.

**Note**: The current sample project includes only the AI Learning Module and backtesting module. Actual trade execution module and Data Feedback/Optimizer Module are planned for future implementation.

### Project Goals

- Provide a sample implementation of AI trading signal generation system
- Demonstrate the concept and implementation of role-based ensemble architecture
- Evaluate model performance through backtesting

### Core Design Principle: Role-Based Ensemble

A structure where each AI model has a clear role and collaborates:

- **XGBoost**: Direction prediction (upward/downward probability)
- **LSTM**: Price target prediction (target price, stop loss)
- **Reinforcement Learning**: Position size and timing decisions

## System Architecture (Sample Project)

```
┌─────────────────────────────────────────────────────────┐
│  Data Collection & Feature Engineering                   │
│  - Market data collection (Yahoo Finance)                │
│  - Technical indicator generation                        │
│    (MA, MACD, RSI, Bollinger Bands, etc.)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  AI Learning Module                                     │
│  - XGBoost: Direction prediction                        │
│  - LSTM: Price prediction                               │
│  - Reinforcement Learning: Final action decision        │
│  - Role-based ensemble                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Backtesting Module                                     │
│  - Trade simulation with historical data                 │
│  - Performance metrics calculation                       │
│    (returns, win rate, Sharpe Ratio, etc.)              │
└─────────────────────────────────────────────────────────┘

Note: Actual trade execution module and Data Feedback/Optimizer Module are planned for future implementation.
```

## Project Structure

```
role-based-trading-ai/
├── README.md                          # Project overview (this document)
├── LICENSE                            # MIT License
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Git ignore file
├── docs/                              # Detailed technical documentation
│   ├── architecture.md               # AI model architecture
│   ├── system_design.md              # System design and interfaces
│   └── implementation_guide.md       # Implementation guide
└── example/                           # Sample project
    ├── run_sample.py                  # Sample execution script
    ├── config/                        # Configuration files
    │   └── sample_config.yaml
    ├── data/                          # Data storage
    │   ├── download_data.py          # Data download script
    │   └── nvda_data.csv             # Sample data (NVDA)
    ├── results/                       # Results storage (images, charts, etc.)
    ├── feature_engineering/           # Feature engineering module
    │   └── feature_engineering.py    # Technical indicator generation
    └── ai_learning/                   # AI learning module sample
        ├── models/                    # Model implementations
        │   ├── xgboost_direction.py
        │   ├── lstm_price.py
        │   └── rl_agent.py
        ├── ensemble/                  # Ensemble
        │   └── role_based_ensemble.py
        ├── backtest.py                # Backtesting module
        └── train_sample.py            # Sample training script
```

## Quick Start

### Requirements

- Python 3.8 or higher
- CUDA-enabled GPU (recommended, but works on CPU)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd role-based-trading-ai

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Sample Project

```bash
cd example
python run_sample.py
```

The sample project demonstrates the complete pipeline. It uses NVDA stock data to train XGBoost, LSTM, and Reinforcement Learning models and perform backtesting.

## Key Features

### AI Learning Module

1. **XGBoost Direction Prediction Model**: Direction prediction based on structured features
2. **LSTM Price Prediction Model**: Price target prediction based on time series data
3. **Reinforcement Learning Agent**: Final action and position size decisions
4. **Role-Based Ensemble**: Integrates predictions from each model to generate final signals

## Documentation

### Core Documentation

- **[AI Model Architecture](./docs/architecture.md)**: Detailed explanation of AI model structure and role-based ensemble
- **[System Design](./docs/system_design.md)**: System composition, interfaces, data flow, component design

### Detailed Guides

- **[Implementation Guide](./docs/implementation_guide.md)**: Technology stack, step-by-step implementation methods, GPU/CPU auto-detection

## Technology Stack Summary

- **AI/ML**: XGBoost, PyTorch, Stable-Baselines3, Gymnasium
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Data Collection**: yfinance (Yahoo Finance)
- **Visualization**: Matplotlib
- **Language**: Python 3.8+

**Note**: The sample project does not include MLflow, Prometheus, or Grafana. These are technology stacks that can be used for future expansion.

For detailed technology stack information, please refer to the [Implementation Guide](./docs/implementation_guide.md).

## Important Disclaimer

**This project is provided for educational and research purposes only.**

- This project is **NOT investment advice**. Please perform sufficient validation and testing before trading with real funds.
- Past performance does not guarantee future returns. All investments carry risk of loss.
- Backtest results may differ from actual trading environments (slippage, trading costs, liquidity, etc.).
- The project authors and contributors are not responsible for any losses incurred from using this software.
- Please consult with a professional before using this for actual trading.

## License

This project is distributed under the [MIT License](./LICENSE).

## Contributing

Contributions are welcome! Please note the following:

1. **Issue Reports**: If you find bugs or improvements, please open an issue.
2. **Pull Requests**: 
   - Maintain consistent code style
   - Include tests and documentation when adding new features
   - Write clear commit messages
3. **Code Review**: All PRs go through code review
4. **Documentation**: Update related documentation when changing code

### Development Guidelines

- Python code style: Follow PEP 8
- Type hints recommended
- Write docstrings (Google style recommended)
- Unit tests recommended

## Contact & Support

- Please contact us through the issue tracker
- Bug reports and feature suggestions are welcome

## Acknowledgments

Thank you to everyone who has contributed to this project.

---

**Disclaimer**: This software is provided "as is", without warranty of any kind, express or implied. The authors and contributors are not liable for any loss or damage arising from the use of this software.

