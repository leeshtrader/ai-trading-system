# 구현 가이드

## 개요

본 문서는 **샘플 프로젝트**의 AI Learning Module 구현 방법을 단계별로 안내합니다.

**참고**: 현재 샘플 프로젝트에는 AI Learning Module만 구현되어 있으며, Data Feedback/Optimizer Module은 향후 구현 예정입니다.

## 기술 스택

### AI/ML 프레임워크

- **XGBoost**: 구조화된 데이터 기반 방향 예측 (GPU 자동 감지 지원)
- **PyTorch**: LSTM 구현 및 강화학습 (GPU 자동 감지 지원)
- **Stable-Baselines3**: 강화학습 (PPO, GPU 자동 감지 지원)
- **Optuna**: 하이퍼파라미터 최적화 (보상 파라미터 최적화에 사용)

### 데이터 처리

- **Pandas/NumPy**: 데이터 처리
- **Scikit-learn**: 전처리 및 평가
- **Statsmodels**: 시계열 분석

### 설정 관리

- **PyYAML**: YAML 설정 파일 파싱 및 관리
- **config_loader.py**: 설정 파일 로드 및 접근 유틸리티

### 모니터링 및 관리 (향후 확장)

- **MLflow**: 모델 버전 관리 및 실험 추적 (향후 구현)
- **Prometheus**: 시스템 메트릭 수집 (향후 구현)
- **Grafana**: 실시간 대시보드 (향후 구현)

### 데이터 저장

- **파일 기반 저장**: CSV, JSON 파일로 저장
- **직접 DB 구현**: 필요시 직접 구현

### GPU/CPU 자동 감지

모든 모델(XGBoost, LSTM, RL Agent)은 자동으로 GPU를 감지하여 사용합니다:
- **XGBoost**: `torch.cuda.is_available()`로 GPU 감지 후 `tree_method='gpu_hist'` 설정
- **LSTM**: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`로 자동 선택
- **RL Agent**: Stable-Baselines3의 `device='cuda'` 파라미터로 GPU 사용

GPU가 없는 경우 자동으로 CPU로 전환되므로 별도 설정이 필요 없습니다.

자세한 패키지 목록은 `requirements.txt`를 참고하세요.

## 구현 단계별 가이드

### Phase 1: 환경 설정

#### 1.1 프로젝트 구조 (샘플 프로젝트 기준)

현재 샘플 프로젝트 구조:
```
role-based-trading-ai/
├── example/
│   ├── run_sample.py                  # 전체 파이프라인 실행
│   ├── inference.py                   # 추론 스크립트
│   ├── config/
│   │   ├── sample_config.yaml         # YAML 설정 파일
│   │   └── config_loader.py           # 설정 로더 유틸리티
│   ├── data/
│   │   ├── download_data.py          # 데이터 다운로드
│   │   └── nvda_data.csv             # 샘플 데이터
│   ├── models/                        # 학습된 모델 저장소
│   │   ├── xgboost_model.pkl
│   │   ├── lstm_model.pth
│   │   └── rl_agent.zip
│   ├── results/                       # 결과 저장소
│   │   ├── xgboost/                   # XGBoost 결과
│   │   ├── lstm/                      # LSTM 결과
│   │   ├── lstm_comparison/           # LSTM 비교 결과
│   │   ├── rl/                        # 강화학습 결과
│   │   ├── diagnosis/                 # 데이터 진단
│   │   └── optimization/             # 최적화 결과
│   ├── tools/                         # 최적화 도구
│   │   ├── optimize_thresholds.py    # 임계값 최적화
│   │   └── optimize_reward_params.py  # 보상 파라미터 최적화
│   ├── feature_engineering/
│   │   └── feature_engineering.py
│   └── ai_learning/
│       ├── models/
│       │   ├── xgboost_direction.py
│       │   ├── lstm_price.py
│       │   └── rl_agent.py
│       ├── ensemble/
│       │   └── role_based_ensemble.py
│       ├── backtest.py
│       ├── train_sample.py            # 전체 파이프라인 학습
│       ├── train_xgboost.py           # XGBoost 개별 학습
│       └── train_lstm.py              # LSTM 개별 학습
├── docs/
└── requirements.txt
```

새 프로젝트를 시작하는 경우:
```bash
mkdir -p example/{ai_learning,feature_engineering,data,config,tools,models,results}
mkdir -p example/ai_learning/{models,ensemble}
mkdir -p example/results/{xgboost,lstm,lstm_comparison,rl,diagnosis,optimization}
```

#### 1.2 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 1.3 설정 파일 생성 및 사용

샘플 프로젝트에서는 `example/config/sample_config.yaml` 파일을 사용합니다.
설정 파일은 `config_loader.py`를 통해 로드됩니다.

```python
# 설정 파일 사용 예시
from config.config_loader import load_config, get_data_config, get_model_config

# 설정 파일 로드
config = load_config()

# 특정 섹션 가져오기
data_config = get_data_config(config)
xgboost_config = get_model_config(config, 'xgboost')
```

설정 파일 구조 (sample_config.yaml 참고):
```yaml
data:
  symbol: "NVDA"
  start_date: "auto"  # 자동 계산 (years_back 사용)
  end_date: "auto"    # 오늘 날짜
  years_back: 5

models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.05
  
  lstm:
    sequence_length: 20
    lstm_units: 32
    dropout_rate: 0.2
  
  rl:
    algorithm: PPO
    learning_rate: 3e-4
    total_timesteps: 100000
    gamma: 0.99

training:
  train_test_split: 0.8
  validation_split: 0.8

backtest:
  initial_balance: 100000
  transaction_cost_rate: 0.001
```

### Phase 2: XGBoost 모델 구현

#### 2.1 기본 모델 구조

```python
# ai_learning/models/xgboost_direction.py
import xgboost as xgb
import numpy as np
from typing import Dict, List
import joblib

class XGBoostDirectionModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        # GPU 자동 감지
        import torch
        self.use_gpu = torch.cuda.is_available()
    
    def build_model(self):
        """모델 구조 생성 (GPU 자동 감지)"""
        params = {
            'n_estimators': self.config['n_estimators'],
            'max_depth': self.config['max_depth'],
            'learning_rate': self.config['learning_rate'],
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': 3,  # BUY, SELL, HOLD
            'random_state': 42
        }
        
        # GPU가 있으면 GPU 사용, 없으면 CPU
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
        else:
            params['tree_method'] = 'hist'
        
        self.model = xgb.XGBClassifier(**params)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """모델 학습"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=True
        )
    
    def predict(self, features: np.ndarray) -> Dict:
        """방향 예측"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(features)[0]
        
        return {
            'buy_prob': float(proba[0]),
            'sell_prob': float(proba[1]),
            'hold_prob': float(proba[2]),
            'trend_strength': float(abs(proba[0] - proba[1])),
            'confidence': float(max(proba))
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """모델 로드"""
        self.model = joblib.load(filepath)
```

#### 2.2 학습 스크립트 (샘플 프로젝트 참고)

실제 샘플 프로젝트에서는 `example/ai_learning/train_sample.py`를 참고하세요.
해당 스크립트는 전체 파이프라인(XGBoost, LSTM, RL)을 포함하고 있습니다.

### Phase 3: LSTM 모델 구현

#### 3.1 기본 모델 구조 (GPU/CPU 자동 감지 포함)

```python
# ai_learning/models/lstm_price.py
import torch
import torch.nn as nn
from typing import Dict, Tuple

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # target_price, stop_loss, time_horizon, confidence, volatility
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 마지막 timestep만 사용
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

class LSTMPriceModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        # CPU/GPU 자동 감지 (각 모델에서 직접 처리)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self, input_size: int):
        """모델 구조 생성"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config['lstm_units'],
            num_layers=3,
            dropout=self.config['dropout_rate']
        ).to(self.device)  # GPU 자동 감지 후 배치 (없으면 CPU)
    
    def train(self, train_loader, val_loader, epochs: int = 50):
        """모델 학습"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate(val_loader, criterion)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    def predict(self, sequences: torch.Tensor) -> Dict:
        """가격 예측"""
        self.model.eval()
        with torch.no_grad():
            sequences = sequences.to(self.device)
            outputs = self.model(sequences)
            outputs = outputs.cpu().numpy()[0]
        
        return {
            'target_price': float(outputs[0]),
            'stop_loss': float(outputs[1]),
            'time_horizon': int(outputs[2]),
            'price_confidence': float(outputs[3]),
            'volatility_forecast': float(outputs[4])
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath: str, input_size: int):
        """모델 로드"""
        self.build_model(input_size)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
```

### Phase 4: 강화학습 Agent 구현

#### 4.1 환경 정의

#### 4.1 환경 정의 (샘플 프로젝트 참고)

실제 샘플 프로젝트에서는 `example/ai_learning/models/rl_agent.py`에 `SimpleTradingEnv`가 구현되어 있습니다.
다음은 참고용 예시입니다:

```python
# ai_learning/models/trading_env.py (참고용)
import gymnasium as gym  # 또는 gym
from gymnasium import spaces  # 또는 from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, market_data, initial_balance=100000):
        super(TradingEnv, self).__init__()
        
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # -1.0 ~ 1.0
        self.current_step = 0
        
        # Action space: [action_type, position_size]
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # position_size: 0.0 ~ 1.0
        self.action_space = spaces.Box(
            low=np.array([0, 0.0]),
            high=np.array([2, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [direction_signal, price_target, current_state]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # 조정 필요
            dtype=np.float32
        )
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        action_type = int(action[0])
        position_size = float(action[1])
        
        current_price = self.market_data[self.current_step]['price']
        
        # 행동 실행
        if action_type == 1:  # BUY
            new_position = min(self.position + position_size, 1.0)
        elif action_type == 2:  # SELL
            new_position = max(self.position - position_size, -1.0)
        else:  # HOLD
            new_position = self.position
        
        # 다음 스텝으로 이동
        self.current_step += 1
        next_price = self.market_data[self.current_step]['price']
        
        # 수익률 계산
        price_change = (next_price - current_price) / current_price
        reward = self.position * price_change
        
        # 거래 비용 차감
        if action_type != 0:
            reward -= 0.001 * abs(position_size)
        
        # 포지션 업데이트
        self.position = new_position
        
        # 종료 조건
        done = (self.current_step >= len(self.market_data) - 1) or (self.balance <= 0)
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """관측값 생성"""
        # 실제 구현에서는 XGBoost/LSTM 예측 결과를 포함
        return np.array([
            self.position,
            self.balance / self.initial_balance,
            # ... 기타 상태 정보
        ])
```

#### 4.2 RL Agent 구현

```python
# ai_learning/models/rl_agent.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict

class TradingRLAgent:
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        self.agent = None
        # GPU 자동 감지
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def build_agent(self):
        """Agent 생성 (GPU 자동 감지)"""
        self.agent = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config['gamma'],
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            device=self.device,  # GPU 자동 감지
            verbose=1
        )
    
    def train(self, total_timesteps: int = 100000):
        """학습"""
        if self.agent is None:
            self.build_agent()
        
        callback = PerformanceCallback()
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
    
    def predict(self, state: Dict):
        """행동 예측"""
        obs = self._state_to_observation(state)
        action, _ = self.agent.predict(obs, deterministic=True)
        return self._action_to_signal(action)
    
    def _state_to_observation(self, state: Dict):
        """상태를 관측 벡터로 변환"""
        # 구현 생략
        pass
    
    def _action_to_signal(self, action):
        """행동을 신호로 변환"""
        return {
            'action_type': int(action[0]),
            'position_size': float(action[1])
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """모델 로드"""
        self.agent = PPO.load(filepath, env=self.env)

class PerformanceCallback(BaseCallback):
    def __init__(self):
        super(PerformanceCallback, self).__init__()
    
    def _on_step(self):
        # 성과 모니터링
        return True
```

### Phase 5: 역할 분담 앙상블 구현

#### 5.1 역할 분담 앙상블 구현 (실제 구현 참고)

실제 샘플 프로젝트의 구현은 `example/ai_learning/ensemble/role_based_ensemble.py`를 참고하세요.
다음은 기본 구조 예시입니다:

```python
# ai_learning/ensemble/role_based_ensemble.py (참고용)
from typing import Dict
import numpy as np

class RoleBasedEnsemble:
    def __init__(self, xgb_model, lstm_model, rl_agent):
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.rl = rl_agent
    
    def generate_signal(self, features, sequences, current_state: Dict) -> Dict:
        """최종 거래 신호 생성"""
        # 1. XGBoost 방향 예측
        direction_pred = self.xgb.predict(features)
        
        # 2. LSTM 가격 예측 (현재 가격 전달)
        current_price = current_state.get('current_price', 150.0)
        price_pred = self.lstm.predict(sequences, current_price=current_price)
        
        # 3. 강화학습 최종 결정
        state = {
            'direction': direction_pred,
            'price_target': price_pred,
            'current_state': current_state
        }
        
        rl_action = self.rl.predict(state)
        
        # 4. 최종 신호 구성
        return {
            'action': self._action_type_to_string(rl_action['action_type']),
            'size': rl_action['position_size'],
            'target_price': price_pred['target_price'],
            'stop_loss': price_pred['stop_loss'],
            'confidence': self._calculate_confidence(
                direction_pred, price_pred, rl_action
            ),
            'reasoning': {
                'direction_signal': direction_pred,
                'price_prediction': price_pred,
                'rl_decision': rl_action
            }
        }
    
    def _action_type_to_string(self, action_type: int) -> str:
        """행동 타입을 문자열로 변환"""
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(action_type, 'HOLD')
    
    def _calculate_confidence(self, direction, price, rl_action):
        """신뢰도 계산"""
        direction_conf = max(direction.get('buy_prob', 0), direction.get('sell_prob', 0))
        price_conf = price.get('price_confidence', 0.5)
        rl_conf = rl_action.get('confidence', 0.5)
        
        return (
            direction_conf * 0.3 +
            price_conf * 0.4 +
            rl_conf * 0.3
        )
```

### Phase 6: 백테스트 모듈 (실제 구현됨)

현재 샘플 프로젝트에는 백테스트 모듈이 구현되어 있습니다:
- 위치: `example/ai_learning/backtest.py`
- 기능: 학습된 모델들을 사용하여 과거 데이터로 거래 시뮬레이션 수행

### 향후 구현 예정: Data Feedback/Optimizer Module

**참고**: 현재 샘플 프로젝트에는 Data Feedback/Optimizer Module이 구현되어 있지 않습니다. 이는 향후 구현 예정 기능입니다.

계획된 구현 예시:

```python
# optimizer/feedback_loop.py (향후 구현)
from typing import List, Dict
import pandas as pd
import numpy as np

class FeedbackLoop:
    def __init__(self):
        self.trade_history = []
    
    def collect_trade_result(self, trade_data: Dict):
        """거래 결과 수집"""
        self.trade_history.append(trade_data)
    
    def analyze_performance(self) -> Dict:
        """성과 분석"""
        if not self.trade_history:
            return {}
        
        returns = [trade['pnl'] for trade in self.trade_history]
        
        return {
            'total_return': sum(returns),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': self._calculate_win_rate(returns)
        }
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Sharpe Ratio 계산"""
        if not returns or len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return mean_return / std_return if std_return > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """최대 낙폭 계산"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """승률 계산"""
        if not returns:
            return 0.0
        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)
```

자세한 설계는 [시스템 설계 문서](./system_design.md)를 참고하세요.

## 테스트 가이드

### 단위 테스트 예시 (참고용)

샘플 프로젝트에는 현재 테스트 코드가 포함되어 있지 않습니다. 
다음은 테스트 작성 예시입니다:

```python
# tests/unit_tests/test_xgboost.py (참고용)
import unittest
import numpy as np
import sys
import os

# 샘플 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../example'))
from ai_learning.models.xgboost_direction import XGBoostDirectionModel

class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        }
        self.model = XGBoostDirectionModel(self.config)
    
    def test_predict(self):
        X = np.random.rand(10, 13)  # 실제 피처 개수에 맞춤
        y = np.random.randint(0, 3, 10)
        self.model.train(X, y)
        
        features = np.random.rand(1, 13)
        prediction = self.model.predict(features)
        
        self.assertIn('buy_prob', prediction)
        self.assertIn('sell_prob', prediction)
        self.assertIn('hold_prob', prediction)
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)

if __name__ == '__main__':
    unittest.main()
```

## 추론 및 최적화 도구

### 추론 스크립트 (inference.py)

학습된 모델을 로드하여 새로운 데이터에 대한 예측을 수행합니다.

```bash
cd example
python inference.py
```

주요 기능:
- 저장된 모델 자동 로드 (XGBoost, LSTM, RL Agent)
- 새로운 데이터 다운로드 및 전처리
- 앙상블을 통한 최종 거래 신호 생성
- 예측 결과 출력

### 최적화 도구

#### 1. XGBoost 임계값 최적화 (tools/optimize_thresholds.py)

XGBoost 예측 확률 분포를 분석하여 최적의 buy_threshold, sell_threshold를 계산합니다.

```bash
cd example
python tools/optimize_thresholds.py
```

기능:
- Precision-Recall 곡선 분석
- F1 Score 최적화
- 최적 임계값 계산 및 저장
- 확률 분포 시각화

#### 2. 강화학습 보상 파라미터 최적화 (tools/optimize_reward_params.py)

Optuna를 사용하여 강화학습 보상 함수 파라미터를 Sharpe Ratio 기반으로 최적화합니다.

```bash
cd example
python tools/optimize_reward_params.py
```

기능:
- Optuna 베이지안 최적화
- Sharpe Ratio 목표 최적화
- 최적 파라미터 YAML 파일 저장
- 최적화 과정 시각화

### 개별 모델 학습

#### XGBoost 개별 학습 (train_xgboost.py)

XGBoost 모델만 별도로 학습하고 임계값 최적화를 수행합니다.

```bash
cd example
python ai_learning/train_xgboost.py
```

기능:
- XGBoost 모델 학습
- 자동 임계값 최적화
- 학습 곡선 및 성능 지표 시각화
- 모델 저장

#### LSTM 개별 학습 (train_lstm.py)

LSTM 모델만 별도로 학습하고 성능 평가를 수행합니다.

```bash
cd example
python ai_learning/train_lstm.py
```

기능:
- LSTM 모델 학습
- 가격 예측 성능 평가
- 학습 곡선 및 예측 비교 시각화
- 모델 저장

## 배포 체크리스트

- [ ] 모든 모델 학습 완료
- [ ] 백테스팅 검증 완료
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 성능 테스트 완료
- [ ] 보안 검토 완료
- [ ] 문서화 완료
- [ ] 모니터링 설정 완료
- [ ] 백업 시스템 구축
- [ ] 장애 복구 계획 수립

