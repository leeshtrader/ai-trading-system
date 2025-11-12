# AI 모델 아키텍처

## 개요

본 문서는 **샘플 프로젝트**의 AI Learning Module 아키텍처를 설명합니다.

**참고**: 현재 샘플 프로젝트에는 AI Learning Module만 구현되어 있으며, Data Feedback/Optimizer Module은 향후 구현 예정입니다.

## 핵심 설계: 역할 분담 방식 (Role-Based Ensemble)

각 AI 모델이 명확한 역할을 담당하여 협력하는 구조로, 단순 앙상블보다 해석 가능성과 성능 최적화가 용이합니다.

## AI Learning Module 아키텍처

### 전체 구조도

```
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering                            │
│  (MA, EMA, MACD, RSI, Bollinger Bands, Volume 등)           │
└──────────────────────┬──────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌────────▼────────┐
│  XGBoost       │            │  LSTM           │
│  방향 예측 모델  │            │  가격 예측 모델   │
│                │            │                 │
│ 입력:           │            │ 입력:            │
│ - MA 비율       │            │ - 가격 시퀀스     │
│ - MACD         │            │ - 거래량 시퀀스   │
│ - RSI          │            │ - RSI 시퀀스      │
│ - Bollinger    │            │ - MACD 시퀀스     │
│ - Volume Ratio │            │ - BB Position    │
│ - Volatility   │            │                  │
│                │            │                  │
│ 출력:           │            │ 출력:            │
│ - 매수 확률     │            │ - 목표가          │
│ - 매도 확률     │            │ - 손절가          │
│ - 보유 확률     │            │ - 예상 보유기간    │
│ - 트렌드 강도   │            │ - 가격 신뢰도     │
│ - 신뢰도        │            │ - 변동성 예측     │
└───────┬────────┘            └────────┬────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │  역할 분담 앙상블       │
            │  (신호 융합)           │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  강화학습 Agent        │
            │                       │
            │ 입력:                  │
            │ - 방향 신호            │
            │ - 가격 목표            │
            │ - 현재 포지션          │
            │ - 잔고 상태            │
            │ - 시장 변동성          │
            │                      │
            │ 출력:                 │
            │ - 행동 (매수/매도/보유)│
            │ - 포지션 크기         │
            │ - 신뢰도              │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  최종 거래 신호         │
            │  - action             │
            │  - size               │
            │  - target_price       │
            │  - stop_loss          │
            │  - confidence         │
            └───────────────────────┘
```

### 1. XGBoost 방향 예측 모델

#### 역할
시장의 방향성(상승/하락)을 예측하는 전문 모델

#### 입력 피처
```python
features = {
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
```

#### 출력
```python
direction_prediction = {
    'buy_prob': float,        # 매수 확률 (0.0 ~ 1.0)
    'sell_prob': float,       # 매도 확률 (0.0 ~ 1.0)
    'hold_prob': float,       # 보유 확률 (0.0 ~ 1.0)
    'trend_strength': float,  # 트렌드 강도 (0.0 ~ 1.0)
    'confidence': float       # 신뢰도 (0.0 ~ 1.0)
}
```

### 2. LSTM 가격 예측 모델

#### 역할
가격 목표와 손절가를 예측하는 전문 모델

#### 입력 시퀀스
```python
# LSTM 입력: (n_samples, sequence_length, n_features)
# sequence_length: 20 (기본값)
# n_features: 5
sequences = np.ndarray  # shape: (n_samples, 20, 5)
# 각 시퀀스는 다음 5개 피처를 포함:
# - close: 종가 (정규화됨)
# - volume: 거래량 (정규화됨)
# - rsi: RSI 지표 (정규화됨)
# - macd: MACD 지표 (정규화됨)
# - bb_position: Bollinger Bands 위치 (정규화됨)
```

#### 출력
```python
price_prediction = {
    'target_price': float,      # 목표가
    'stop_loss': float,         # 손절가
    'time_horizon': int,        # 예상 보유 기간 (일)
    'price_confidence': float,  # 가격 예측 신뢰도 (0.0 ~ 1.0)
    'volatility_forecast': float # 예상 변동성
}
```

### 3. 강화학습 Agent

#### 역할
XGBoost와 LSTM의 예측을 종합하여 최종 행동 결정

#### 상태 공간 (State Space)
```python
state = {
    # XGBoost 예측
    'direction': {
        'buy_prob': float,
        'sell_prob': float,
        'trend_strength': float
    },
    # LSTM 예측
    'price_target': {
        'target_price': float,
        'stop_loss': float,
        'time_horizon': int
    },
    # 현재 상태
    'current_position': float,      # 현재 포지션 크기 (-1.0 ~ 1.0)
    'balance': float,                # 잔고
    'unrealized_pnl': float,         # 미실현 손익
    'market_volatility': float,       # 시장 변동성
    'recent_performance': float,     # 최근 성과
    'time_since_entry': int          # 진입 후 경과 시간
}
```

#### 행동 공간 (Action Space)
```python
action = {
    'action_type': int,      # 0: HOLD, 1: BUY, 2: SELL
    'position_size': float,  # 포지션 크기 (0.0 ~ 1.0)
    'urgency': float        # 긴급도 (0.0 ~ 1.0)
}
```

#### 보상 함수 (Reward Function)
```python
def calculate_reward(state, action, next_state):
    """보상 계산"""
    # 1. 수익률 기반 보상
    pnl_reward = (next_state['balance'] - state['balance']) / state['balance']
    
    # 2. 리스크 조정 보상 (Sharpe Ratio 스타일)
    risk_adjusted_reward = pnl_reward / (state['market_volatility'] + 0.01)
    
    # 3. 방향 예측 정확도 보너스
    direction_bonus = 0.0
    if action['action_type'] == 1 and state['direction']['buy_prob'] > 0.6:
        direction_bonus = 0.1
    elif action['action_type'] == 2 and state['direction']['sell_prob'] > 0.6:
        direction_bonus = 0.1
    
    # 4. 손절가 준수 보너스
    stop_loss_bonus = 0.0
    if next_state['unrealized_pnl'] > -state['price_target']['stop_loss']:
        stop_loss_bonus = 0.05
    
    # 5. 거래 비용 페널티
    transaction_cost = -0.001 * abs(action['position_size'])
    
    total_reward = pnl_reward + risk_adjusted_reward + direction_bonus + stop_loss_bonus + transaction_cost
    
    return total_reward
```

### 4. 역할 분담 앙상블

#### 역할
각 모델의 예측을 통합하여 강화학습 Agent에 전달

#### 통합 로직
```python
class RoleBasedEnsemble:
    def generate_signal(self, features, sequences, current_state):
        # 1. XGBoost 방향 예측
        direction_pred = self.xgb_model.predict(features)
        
        # 2. LSTM 가격 예측
        price_pred = self.lstm_model.predict(sequences)
        
        # 3. 강화학습 최종 결정
        state = {
            'direction': direction_pred,
            'price_target': price_pred,
            'current_state': current_state
        }
        
        rl_action = self.rl_agent.predict(state)
        
        # 4. 최종 신호 구성
        signal = {
            'action': rl_action['action_type'],
            'size': rl_action['position_size'],
            'target_price': price_pred['target_price'],
            'stop_loss': price_pred['stop_loss'],
            'confidence': self._calculate_confidence(direction_pred, price_pred, rl_action)
        }
        
        return signal
```

## 백테스트 모듈

### 역할
학습된 모델들을 사용하여 과거 데이터로 거래 시뮬레이션을 수행하고 성과를 평가합니다.

### 주요 기능
- 과거 데이터를 사용한 거래 시뮬레이션
- 수익률, 승률, Sharpe Ratio 등 성과 지표 계산
- 거래 내역 및 포트폴리오 변화 추적

### 구현 위치
- `example/ai_learning/backtest.py`

## 향후 구현 예정: Data Feedback / Optimizer Module

**참고**: 현재 샘플 프로젝트에는 Data Feedback/Optimizer Module이 구현되어 있지 않습니다. 이는 향후 구현 예정 기능입니다.

### 계획된 구조

```
┌─────────────────────────────────────────────────────────┐
│              Strategy Execution                        │
│  (거래 실행 결과)                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Data Feedback / Optimizer Module                      │
│  (향후 구현 예정)                                        │
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

자세한 설계는 [시스템 설계 문서](./system_design.md)와 [구현 가이드](./implementation_guide.md)를 참고하세요.

