# 시스템 설계

## 개요

본 문서는 **샘플 프로젝트**의 AI Learning Module 시스템 설계를 상세히 설명합니다.

**참고**: 현재 샘플 프로젝트에는 AI Learning Module만 구현되어 있으며, Data Feedback/Optimizer Module은 향후 구현 예정입니다.

## 전체 시스템 맥락

```
[데이터 수집] → [AI Learning Module] → [거래 실행] → [Data Feedback/Optimizer Module]
```

## AI Learning Module 설계

### 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│              AI Learning Module                         │
│                                                         │
│  ┌──────────────────────────┐  ┌──────────────────────┐ │
│  │  XGBoost 방향 예측 모델   │  │  LSTM 가격 예측 모델    │ │
│  │                          │  │                      │ │
│  │  입력:                    │  │  입력:               │  │
│  │  - 피처 벡터               │  │  - 시계열 데이터       │  │
│  │    (구조화된 데이터)       │  │                      │  │
│  │                          │  │                      │  │
│  │  출력:                    │  │  출력:               │  │
│  │  - 매수/매도/보유 확률    │  │  - 목표가/손절가        │  │
│  │  - 트렌드 강도            │  │  - 예상 보유기간       │  │
│  │  - 신뢰도                 │  │  - 신뢰도             │  │
│  └──────────┬───────────────┘  └──────────┬───────────┘  │
│             │                               │            │
│             └──────────┬───────────────────┘             │
│                        │                                 │
│             ┌──────────▼──────────────────┐             │
│             │  역할 분담 앙상블             │             │
│             │  - 신호 융합                 │             │
│             │  - 신뢰도 계산                │             │
│             └──────────┬──────────────────┘             │
│                        │                                 │
│             ┌──────────▼──────────────────┐             │
│             │  강화학습 Agent              │             │
│             │  - PPO 알고리즘              │             │
│             │  - 최종 행동 결정             │             │
│             │  - 포지션 크기 결정           │             │
│             └──────────┬──────────────────┘             │
│                        │                                 │
│             ┌──────────▼──────────────────┐             │
│             │  거래 신호 생성               │             │
│             │  - action (BUY/SELL/HOLD)   │             │
│             │  - size (포지션 크기)         │             │
│             │  - target_price             │             │
│             │  - stop_loss                │             │
│             │  - confidence               │             │
│             └──────────┬──────────────────┘             │
│                        │                                 │
│                        ▼                                 │
│              [거래 실행 모듈로 전달]                       │
└─────────────────────────────────────────────────────────┘
```

### 입력 데이터 인터페이스

#### 입력 데이터 형식

```python
# 피처 엔지니어링에서 받는 데이터 구조
class FeatureInput:
    """피처 엔지니어링에서 전달받는 데이터"""
    
    # XGBoost용 구조화된 피처
    features: Dict[str, float] = {
        # 추세 지표
        'ma_5_ratio': float,       # 5일 이동평균 대비 가격 비율
        'ma_20_ratio': float,      # 20일 이동평균 대비 가격 비율
        'ma_60_ratio': float,      # 60일 이동평균 대비 가격 비율
        'macd': float,             # MACD 값
        'macd_signal': float,      # MACD 시그널
        'macd_hist': float,        # MACD 히스토그램
        
        # 평균회귀 지표
        'rsi': float,              # RSI 지표 (0-1로 정규화)
        'bb_position': float,      # Bollinger Bands 위치 (0-1)
        'bb_width': float,         # Bollinger Bands 폭
        'z_score': float,          # Z-Score
        
        # 거래량 및 변동성
        'volume_ratio': float,     # 거래량 비율 (현재/평균)
        'volatility': float,       # 변동성 (20일 롤링 표준편차)
        'price_change': float,     # 가격 변화율
    }
    
    # LSTM용 시계열 데이터
    # shape: (n_samples, sequence_length, n_features)
    # sequence_length: 20 (기본값)
    # n_features: 5 (close, volume, rsi, macd, bb_position)
    sequences: np.ndarray  # shape: (n_samples, 20, 5)
    
    # 현재 상태 정보
    current_state: Dict = {
        'current_price': float,
        'current_position': float,  # -1.0 ~ 1.0
        'balance': float,
        'unrealized_pnl': float,
        'market_volatility': float
    }
```

#### Python 인터페이스

```python
# AI Learning Module에서 제공하는 인터페이스
class AILearningService:
    """AI Learning Module 서비스 인터페이스"""
    
    def generate_trading_signal(self, feature_input: FeatureInput) -> TradingSignal:
        """
        거래 신호 생성
        
        Args:
            feature_input: 피처 엔지니어링에서 전달받는 데이터
        
        Returns:
            TradingSignal: 거래 신호
        """
        pass
```

### 출력 인터페이스

#### 출력 데이터 형식

```python
# 거래 실행 모듈로 전달하는 거래 신호
class TradingSignal:
    """거래 신호"""
    
    action: str  # 'BUY', 'SELL', 'HOLD'
    size: float  # 포지션 크기 (0.0 ~ 1.0)
    target_price: float  # 목표가
    stop_loss: float  # 손절가
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    reasoning: Dict = {
        'direction_signal': Dict,  # XGBoost 예측 결과
        'price_prediction': Dict,  # LSTM 예측 결과
        'rl_decision': Dict        # 강화학습 결정
    }
```

## 백테스트 모듈

### 역할
학습된 모델들을 사용하여 과거 데이터로 거래 시뮬레이션을 수행하고 성과를 평가합니다.

### 구현 위치
- `example/ai_learning/backtest.py`

## 향후 구현 예정: Data Feedback / Optimizer Module 설계

**참고**: 현재 샘플 프로젝트에는 Data Feedback/Optimizer Module이 구현되어 있지 않습니다. 이는 향후 구현 예정 기능입니다.

### 계획된 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│              Strategy Execution                        │
│  (거래 실행 결과)                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Data Feedback / Optimizer Module                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  성과 모니터링                                      │  │
│  │  - 백테스트 결과 (기준선)                           │  │
│  │  - 포워드 테스트 결과                                │  │
│  │  - 성과 비교 및 저하 감지                            │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  피드백 루프 (성과 저하 감지 시에만 활성화)          │  │
│  │  - 거래 결과 수집                                   │  │
│  │  - 성과 분석                                        │  │
│  │  - 모델별 기여도 분석                                │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  전략 최적화                                        │  │
│  │  - 하이퍼파라미터 튜닝                               │  │
│  │  - 모델 가중치 조정                                 │  │
│  │  - 리스크 파라미터 최적화                            │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  모델 재학습                                        │  │
│  │  - 주기적 업데이트                                  │  │
│  │  - 온라인 학습                                      │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│                     ▼                                    │
│              [AI Learning Module로 피드백]                │
└─────────────────────────────────────────────────────────┘
```

### 거래 실행 모듈과의 인터페이스

#### 입력 데이터 형식

```python
# 거래 실행 모듈에서 받는 거래 결과 데이터
class TradeResult:
    """거래 실행 모듈에서 전달받는 거래 결과"""
    
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL'
    entry_price: float
    exit_price: float
    size: float
    pnl: float  # 실현 손익
    
    # AI Learning Module에서 생성한 신호 정보 (분석용)
    original_signal: TradingSignal
    
    # 실행 정보
    execution_info: Dict = {
        'slippage': float,
        'commission': float,
        'execution_time': float
    }
```

#### Python 인터페이스

```python
# Data Feedback/Optimizer Module에서 제공하는 인터페이스
class OptimizerService:
    """Optimizer Module 서비스 인터페이스"""
    
    def record_trade_result(self, trade_result: TradeResult):
        """거래 결과 기록"""
        pass
    
    def get_performance_metrics(self) -> Dict:
        """성과 지표 조회"""
        pass
```

### AI Learning Module과의 인터페이스

#### 출력 데이터 형식

```python
# AI Learning Module로 전달하는 최적화 결과
class OptimizationResult:
    """AI Learning Module로 전달하는 최적화 결과"""
    
    # 업데이트된 모델 파라미터
    model_updates: Dict = {
        'xgboost': {
            'hyperparameters': Dict,
            'model_path': str,
            'version': str
        },
        'lstm': {
            'hyperparameters': Dict,
            'model_path': str,
            'version': str
        },
        'rl_agent': {
            'hyperparameters': Dict,
            'model_path': str,
            'version': str
        }
    }
    
    # 최적화된 가중치
    ensemble_weights: Dict = {
        'xgb_weight': float,
        'lstm_weight': float,
        'rl_weight': float
    }
    
    # 성과 개선 정보
    performance_improvement: Dict = {
        'previous_sharpe': float,
        'current_sharpe': float,
        'improvement': float
    }
```

## 데이터 저장소 설계

### Data Feedback/Optimizer Module 데이터 저장소

#### 파일 기반 저장

```python
# 파일로 저장 (CSV, JSON)
class FileStorage:
    """파일 기반 저장"""
    
    def save_trade_result(self, trade_result: TradeResult):
        """CSV 파일로 저장"""
        pass
```

#### 직접 DB 구현

필요시 직접 구현

### AI Learning Module 데이터 저장소

```python
# 모델 메타데이터 저장 (MLflow - 로컬 파일 시스템 또는 서버)
# - 모델 버전
# - 학습 파라미터
# - 성능 지표
# - 모델 아티팩트
# 
# MLflow는 로컬 파일 시스템에 저장 가능하므로 DB 불필요
```

## 컴포넌트 설계

### AI Learning Module 컴포넌트 (실제 구현)

1. **XGBoost 모델**: `example/ai_learning/models/xgboost_direction.py`
2. **LSTM 모델**: `example/ai_learning/models/lstm_price.py`
3. **강화학습 Agent**: `example/ai_learning/models/rl_agent.py`
4. **역할 분담 앙상블**: `example/ai_learning/ensemble/role_based_ensemble.py`
5. **백테스트 모듈**: `example/ai_learning/backtest.py`
6. **피처 엔지니어링**: `example/feature_engineering/feature_engineering.py`

### Data Feedback/Optimizer Module 컴포넌트 (향후 구현 예정)

**참고**: 현재 샘플 프로젝트에는 구현되어 있지 않습니다.

1. **성과 모니터링**: `optimizer/performance_monitor.py` (향후 구현)
2. **피드백 루프**: `optimizer/feedback_loop.py` (향후 구현)
3. **전략 최적화**: `optimizer/optimizer.py` (향후 구현)

## 인터페이스 요약

### AI Learning Module 제공 인터페이스

- `generate_trading_signal(feature_input) -> TradingSignal`: 거래 신호 생성

### Data Feedback/Optimizer Module 제공 인터페이스

- `record_trade_result(trade_result)`: 거래 결과 기록
- `get_performance_metrics() -> Dict`: 성과 지표 조회
- `should_trigger_feedback_loop() -> bool`: 피드백 루프 활성화 여부

### Data Feedback/Optimizer Module 요구 인터페이스

- 파일 저장 인터페이스
- 또는 직접 DB 구현

