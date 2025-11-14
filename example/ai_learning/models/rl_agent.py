"""
샘플: 강화학습 Agent
강화학습 모델 샘플 구현

주의: 본 모듈은 샘플/데모 목적입니다.

Sample: Reinforcement Learning Agent
Sample implementation of RL agent for trading decisions.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import numpy as np
from typing import Dict, Optional
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용

# stable-baselines3 import 시도
STABLE_BASELINES3_AVAILABLE = False
RL_ALGORITHMS = {}
try:
    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
    from stable_baselines3.common.env_util import make_vec_env
    # 지원하는 알고리즘 딕셔너리
    RL_ALGORITHMS = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN,
        'SAC': SAC,
        'TD3': TD3,
        'DDPG': DDPG
    }
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    # 더미 클래스 (에러 방지)
    class DummyAlgorithm:
        def __init__(self, *args, **kwargs):
            pass
        def learn(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return [0, 0.0], None
        def save(self, *args, **kwargs):
            pass
        @staticmethod
        def load(*args, **kwargs):
            return None
    
    for alg_name in ['PPO', 'A2C', 'DQN', 'SAC', 'TD3', 'DDPG']:
        RL_ALGORITHMS[alg_name] = DummyAlgorithm

# gym/gymnasium import 시도
GYM_AVAILABLE = False
gym = None
spaces = None
try:
    # 먼저 gymnasium 시도 (최신 권장)
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        # gymnasium이 없으면 gym 시도
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        # 둘 다 없을 경우 더미 클래스
        class gym:
            class Env:
                pass
        
        class spaces:
            Box = None
        
        if STABLE_BASELINES3_AVAILABLE:
            print("[샘플] gym/gymnasium이 설치되지 않았습니다.")
            print("[샘플] gymnasium을 설치하세요: pip install gymnasium")
        else:
            print("[샘플] stable-baselines3가 설치되지 않았습니다.")
            print("[샘플] 강화학습 기능은 제한적으로 동작합니다.")

# Trading Environment 클래스
if GYM_AVAILABLE and STABLE_BASELINES3_AVAILABLE:
    class SimpleTradingEnv(gym.Env):
        """
        샘플: 간단한 트레이딩 환경
        강화학습을 위한 환경 정의
        """
        
        def __init__(self, price_data: np.ndarray, initial_balance: float = 100000, 
                     config: Dict = None, xgb_model=None, lstm_model=None, 
                     features_data=None, sequences_data=None):
            super(SimpleTradingEnv, self).__init__()
            
            if config is None:
                config = {}
            
            self.price_data = price_data
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.position = 0.0  # -1.0 ~ 1.0
            self.entry_price = 0.0  # 진입가 초기화
            self.current_step = 0
            self.config = config
            
            # XGBoost/LSTM 모델 저장 (학습 시 observation에 포함)
            self.xgb_model = xgb_model
            self.lstm_model = lstm_model
            self.features_data = features_data  # XGBoost용 피처 데이터
            self.sequences_data = sequences_data  # LSTM용 시퀀스 데이터
            
            # Action Space: Box(1) - [position_size]만 사용
            # 주식은 숏 포지션이 없으므로 position_size만으로 충분
            # position_size: 0.0 (청산) ~ 1.0 (최대 롱)
            # position_size가 증가하면 매수, 감소하면 매도
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            )
            
            # Observation: [position, balance_ratio, price_normalized, price_change, ..., xgb_prob, lstm_target, ...]
            observation_space_size = config.get('observation_space_size', 14)  # XGBoost/LSTM 정보 포함
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_space_size,),
                dtype=np.float32
            )
        
        def reset(self, seed=None, options=None):
            # gymnasium v0.28+ API 호환성
            self.balance = self.initial_balance
            self.position = 0.0
            self.entry_price = 0.0  # 진입가 초기화
            self.current_step = 0
            observation = self._get_observation()
            info = {}
            return observation, info
        
        def step(self, action):
            # gymnasium v0.28+ API: (observation, reward, terminated, truncated, info)
            # Action: [position_size] - 주식은 position_size만 사용
            # position_size: 0.0 (청산) ~ 1.0 (최대 롱)
            # action을 1차원 배열로 변환하고 스칼라로 안전하게 추출
            action = np.asarray(action).flatten()
            action = np.clip(action, 0.0, 1.0)  # 안전장치
            
            # position_size 추출 및 이산화 (10% 단위로만 선택 가능)
            if len(action) > 0:
                position_size_val = action[0]
                if hasattr(position_size_val, '__len__') and not isinstance(position_size_val, str):
                    position_size = float(np.asarray(position_size_val).flatten()[0])
                else:
                    position_size = float(position_size_val)
            else:
                position_size = 0.0
            
            position_size = float(np.clip(position_size, 0.0, 1.0))
            
            # 액션 이산화: 10% 단위로만 선택 가능 (0.0, 0.1, 0.2, ..., 1.0)
            # 이렇게 하면 미세한 조절(10% 미만)을 아예 할 수 없게 됨
            min_meaningful_change = self.config.get('min_meaningful_position_change', 0.10)
            position_size = round(position_size / min_meaningful_change) * min_meaningful_change
            position_size = float(np.clip(position_size, 0.0, 1.0))
            
            # action_type 제거: position_size만 사용
            
            if self.current_step >= len(self.price_data) - 1:
                observation = self._get_observation()
                reward = 0.0
                terminated = True
                truncated = False
                info = {}
                return observation, reward, terminated, truncated, info
            
            # 가격을 스칼라로 안전하게 변환
            current_price_val = self.price_data[self.current_step]
            if hasattr(current_price_val, '__len__') and not isinstance(current_price_val, str):
                current_price = float(np.asarray(current_price_val).flatten()[0])
            else:
                current_price = float(current_price_val)
            
            next_price_val = self.price_data[self.current_step + 1]
            if hasattr(next_price_val, '__len__') and not isinstance(next_price_val, str):
                next_price = float(np.asarray(next_price_val).flatten()[0])
            else:
                next_price = float(next_price_val)
            
            # position을 스칼라로 안전하게 변환 (먼저 정의)
            position_val = self.position
            if hasattr(position_val, '__len__') and not isinstance(position_val, str):
                position_scalar = float(np.asarray(position_val).flatten()[0])
            else:
                position_scalar = float(position_val)
            
            # 행동 실행: position_size를 직접 목표 포지션으로 설정
            # position_size가 증가하면 매수, 감소하면 매도
            new_position_val = position_size
            
            # 포지션이 0에서 양수로 변경되면 진입가 설정
            if position_scalar == 0.0 and new_position_val > 0.0:
                self.entry_price = current_price
            
            # 포지션이 0이 되면 진입가 초기화
            if position_scalar != 0.0 and new_position_val == 0.0:
                self.entry_price = 0.0
            
            # new_position을 스칼라로 보장
            if hasattr(new_position_val, '__len__') and not isinstance(new_position_val, str):
                new_position = float(np.asarray(new_position_val).flatten()[0])
            else:
                new_position = float(new_position_val)
            
            # 추가 매수 시 평균 진입가 업데이트 (분할매수 지원)
            # 포지션이 증가하고 기존 포지션이 있을 때 평균 진입가 계산
            if position_scalar > 0.0 and new_position > position_scalar:
                # 추가 매수량 계산
                additional_size = new_position - position_scalar
                # 평균 진입가 = (기존 포지션 가치 + 추가 매수 가치) / 총 포지션
                total_value = self.entry_price * position_scalar + current_price * additional_size
                self.entry_price = total_value / new_position if new_position > 0.0 else current_price
            
            # 수익률 계산
            price_change = (next_price - current_price) / current_price if current_price > 0 else 0.0
            
            # 설정 파일에서 보상 파라미터 읽기
            reward_scale = self.config.get('reward_scale', 200)
            opportunity_cost_penalty = self.config.get('opportunity_cost_penalty', 50)
            avoidance_bonus = self.config.get('avoidance_bonus', 30)
            correct_direction_bonus = self.config.get('correct_direction_bonus', 100)
            wrong_direction_penalty = self.config.get('wrong_direction_penalty', 50)
            action_bonus = self.config.get('action_bonus', 0.5)
            
            # 거래 비용 계산 (백테스트와 동일한 방식)
            # 백테스트: transaction_cost = abs(position) * balance * transaction_cost_rate
            # RL 환경: transaction_cost = abs(position_change) * balance * transaction_cost_rate
            # 백테스트 설정에서 transaction_cost_rate 읽기
            backtest_config = self.config.get('backtest', {})
            transaction_cost_rate = backtest_config.get('transaction_cost_rate', 0.001)  # 0.1%
            
            # 기본 보상: 포지션 크기를 고려한 수익률 기반 보상 (논리적 일관성 개선)
            # 실제 수익 = 가격 변화율 × 포지션 크기
            # 포지션 크기가 클수록 실제 수익이 커지므로, 보상도 포지션 크기에 비례해야 함
            if position_scalar != 0.0:
                # 실제 수익률 = 가격 변화율 × 포지션 크기
                actual_return = price_change * position_scalar
                reward = float(actual_return * reward_scale)
                
                # 리스크 조정: 변동성 페널티는 포지션 크기의 제곱에 비례 (분산의 성질)
                # 포지션 크기가 2배가 되면 리스크는 4배가 됨 (분산 = 표준편차^2)
                risk_penalty_scale = self.config.get('position_risk_penalty_scale', 80)
                # 변동성 추정 (최근 가격 변화의 절대값)
                volatility_estimate = abs(price_change)
                # 포지션 크기의 제곱에 비례하는 리스크 페널티
                position_risk_penalty = (position_scalar ** 2) * volatility_estimate * risk_penalty_scale
                reward = float(reward - position_risk_penalty)
            else:
                # 포지션 0 유지에 대한 보상: 기회비용 회피 보너스
                # 가격이 하락하면 보너스 (손실 회피), 상승하면 작은 페널티 (기회비용)
                small_opportunity_cost_penalty = self.config.get('opportunity_cost_penalty', 50) * 0.1  # 기회비용의 10%
                if price_change < 0:
                    # 손실 회피 보너스: 가격 하락 시 포지션 0 유지는 올바른 결정
                    reward = float(abs(price_change) * avoidance_bonus)
                else:
                    # 작은 기회비용 페널티: 가격 상승 시 포지션 0 유지는 기회비용
                    reward = float(-price_change * small_opportunity_cost_penalty)
            
            # 포지션 변화량 계산 (이산화되어 있으므로 0.0, 0.1, 0.2, ... 같은 이산값만 나옴)
            position_change = abs(new_position - position_scalar)
            
            # 거래 비용 계산 (백테스트와 동일한 방식: 포지션 변경량 × 자본금 × 수수료율)
            # 포지션이 실제로 변경되었을 때만 거래 비용 발생
            if position_change > 0.0:  # 이산화되어 있으므로 정확한 0.0 비교 가능
                # 백테스트와 동일: transaction_cost = abs(position_change) * balance * transaction_cost_rate
                # position_change는 비중(0.0~1.0), balance는 실제 자본금
                transaction_cost = position_change * self.balance * transaction_cost_rate
                reward = float(reward - transaction_cost)
            
            # 포지션 유지/전체 청산에 대한 기회비용 페널티
            # 포지션이 0이 되면 (유지 또는 전체 청산) 다음 스텝에서 가격 상승 시 기회비용 발생
            if new_position == 0.0:  # 포지션이 0이 됨 (유지 또는 전체 청산)
                if price_change > 0.01:  # 가격이 1% 이상 오르면
                    # 전체 청산 시에는 청산한 포지션 크기에 비례하여 페널티 적용
                    # 포지션 유지(0→0)는 작은 페널티, 전체 청산(1.0→0.0)은 큰 페널티
                    penalty_multiplier = 1.0 if position_scalar == 0.0 else position_scalar  # 전체 청산 시 포지션 크기만큼 페널티
                    reward = float(reward - abs(price_change) * opportunity_cost_penalty * penalty_multiplier)
                elif price_change < -0.01:  # 가격이 1% 이상 떨어지면
                    # 손실 회피 보너스 (포지션 유지 또는 전체 청산으로 손실 회피)
                    bonus_multiplier = 1.0 if position_scalar == 0.0 else position_scalar  # 전체 청산 시 포지션 크기만큼 보너스
                    reward = float(reward + abs(price_change) * avoidance_bonus * bonus_multiplier)
            
            # 포지션 증가/감소에 대한 방향성 보상
            if new_position > position_scalar:  # 포지션 증가
                if price_change > 0:
                    reward = float(reward + abs(price_change) * correct_direction_bonus * position_change)
                else:
                    reward = float(reward + price_change * wrong_direction_penalty * position_change)
            elif new_position < position_scalar:  # 포지션 감소
                if price_change < 0:
                    reward = float(reward + abs(price_change) * correct_direction_bonus * position_change)
                else:
                    reward = float(reward + abs(price_change) * wrong_direction_penalty * position_change)
            
            # 포지션 조절 보상 (수익이 있을 때 포지션을 줄이면 보상, 손실이 있을 때 포지션을 늘리면 페널티)
            # 주의: entry_price는 위에서 이미 평균 진입가로 업데이트되었으므로, 평가손익은 정확하게 계산됨
            if position_scalar != 0.0 and new_position != position_scalar:
                # 업데이트된 평균 진입가 기준으로 평가손익 계산
                unrealized_pnl_ratio = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
                position_change_abs = abs(new_position - position_scalar)  # 절대 변화량
                
                # 수익이 있을 때 포지션을 줄이면 보상 (익실현) - 변화량이 클수록 보상
                profit_taking_scale = self.config.get('profit_taking_bonus_scale', 200)  # reward_scale과 동일
                loss_expansion_scale = self.config.get('loss_expansion_penalty_scale', 100)  # reward_scale의 절반
                if unrealized_pnl_ratio > 0.01 and new_position < position_scalar:  # 1% 이상 수익 + 포지션 감소
                    profit_taking_bonus = unrealized_pnl_ratio * position_change_abs * profit_taking_scale
                    reward = float(reward + profit_taking_bonus)
                # 손실이 있을 때 포지션을 늘리면 페널티 (손실 확대) - 단, 강한 상승 예상 시 보너스/페널티 감소
                elif unrealized_pnl_ratio < -0.01 and new_position > position_scalar:  # 1% 이상 손실 + 포지션 증가
                    # XGBoost/LSTM 신호 확인 (강한 상승 예상 여부)
                    strong_buy_signal = False
                    buy_prob = 0.5
                    target_price_ratio = 1.0
                    
                    # XGBoost 신호 확인
                    if self.xgb_model is not None and self.features_data is not None and self.current_step < len(self.features_data):
                        try:
                            features = self.features_data.iloc[self.current_step].values.reshape(1, -1)
                            direction_pred = self.xgb_model.predict_proba(features)[0]
                            direction_pred = np.asarray(direction_pred).flatten()
                            if len(direction_pred) >= 3:
                                buy_prob_val = direction_pred[1]
                                if hasattr(buy_prob_val, '__len__') and not isinstance(buy_prob_val, str):
                                    buy_prob = float(np.asarray(buy_prob_val).flatten()[0])
                                else:
                                    buy_prob = float(buy_prob_val)
                        except:
                            pass
                    
                    # LSTM 목표가 확인
                    if self.lstm_model is not None and self.sequences_data is not None and self.current_step < len(self.sequences_data):
                        try:
                            sequence = self.sequences_data[self.current_step].reshape(1, -1, 5)
                            price_pred = self.lstm_model.predict(sequence, current_price=current_price)
                            target_price_val = price_pred.get('target_price', current_price * 1.02)
                            target_price = float(np.asarray(target_price_val).flatten()[0]) if hasattr(target_price_val, '__len__') else float(target_price_val)
                            target_price_ratio = target_price / current_price if current_price > 0 else 1.02
                        except:
                            pass
                    
                    # 강한 상승 예상 판단: XGBoost BUY 확률이 높거나 LSTM 목표가가 현재가 대비 충분히 높을 때
                    # BUY 확률 > 0.6 또는 목표가가 현재가 대비 3% 이상 높으면 강한 상승 예상
                    strong_buy_signal = (buy_prob > 0.6) or (target_price_ratio > 1.03)
                    
                    # 손실 확대 페널티 계산
                    base_penalty = abs(unrealized_pnl_ratio) * position_change_abs * loss_expansion_scale
                    
                    if strong_buy_signal:
                        # 강한 상승 예상 시: 페널티 감소 또는 보너스 부여
                        # XGBoost 신호 강도와 LSTM 목표가 비율에 따라 조정
                        signal_strength = min(1.0, (buy_prob - 0.5) * 2.0)  # 0.5~1.0 → 0.0~1.0
                        target_strength = min(1.0, max(0.0, (target_price_ratio - 1.0) * 10.0))  # 1.0~1.1 → 0.0~1.0
                        combined_strength = (signal_strength + target_strength) / 2.0
                        
                        # 페널티 감소 또는 보너스: 강한 신호일수록 페널티 감소, 매우 강하면 보너스
                        if combined_strength > 0.7:  # 매우 강한 신호
                            # 보너스 부여 (분할매수 장려)
                            averaging_down_bonus = base_penalty * 0.3 * combined_strength  # 페널티의 30%를 보너스로
                            reward = float(reward + averaging_down_bonus)
                        else:  # 강한 신호 (보통)
                            # 페널티 감소
                            penalty_reduction = base_penalty * combined_strength * 0.5  # 페널티의 최대 50% 감소
                            reward = float(reward - (base_penalty - penalty_reduction))
                    else:
                        # 약한 신호 또는 하락 예상: 기존 페널티 적용 (무작정 분할매수 방지)
                        reward = float(reward - base_penalty)
            
            # 최종 reward를 스칼라로 보장
            if hasattr(reward, '__len__') and not isinstance(reward, str):
                reward = float(np.asarray(reward).flatten()[0])
            else:
                reward = float(reward)
            
            # XGBoost/LSTM 신호 일치 보너스 (보상 함수 개선)
            # RL의 결정이 XGBoost/LSTM 신호와 일치하면 추가 보상
            signal_match_bonus = self.config.get('signal_match_bonus', 10.0)
            if self.xgb_model is not None and self.features_data is not None and self.current_step < len(self.features_data):
                try:
                    features = self.features_data.iloc[self.current_step].values.reshape(1, -1)
                    direction_pred = self.xgb_model.predict_proba(features)[0]
                    direction_pred = np.asarray(direction_pred).flatten()  # 1차원 배열로 변환
                    if len(direction_pred) >= 3:
                        # 배열 요소를 스칼라로 안전하게 변환
                        buy_prob_val = direction_pred[1]
                        if hasattr(buy_prob_val, '__len__') and not isinstance(buy_prob_val, str):
                            buy_prob = float(np.asarray(buy_prob_val).flatten()[0])
                        else:
                            buy_prob = float(buy_prob_val)
                        
                        sell_prob_val = direction_pred[2]
                        if hasattr(sell_prob_val, '__len__') and not isinstance(sell_prob_val, str):
                            sell_prob = float(np.asarray(sell_prob_val).flatten()[0])
                        else:
                            sell_prob = float(sell_prob_val)
                        
                        # RL의 결정과 XGBoost 신호 일치 확인 (position_size 기반)
                        # 포지션 증가 + BUY 신호 강함 → 보상
                        if new_position > position_scalar and buy_prob > 0.5:
                            reward = float(reward + signal_match_bonus * buy_prob * position_change)
                        # 포지션 감소 + SELL 신호 강함 → 보상
                        elif new_position < position_scalar and sell_prob > 0.5:
                            reward = float(reward + signal_match_bonus * sell_prob * position_change)
                        # 포지션 유지 + 신호 약함 → 보상
                        elif new_position == position_scalar and (buy_prob < 0.4 and sell_prob < 0.4):
                            reward = float(reward + signal_match_bonus * 0.5)
                except:
                    pass
            
            # LSTM 목표가/손절가 기반 비중 조절 보상
            # 주가가 목표가보다 높으면 비중을 줄이면 보상, 손절가보다 낮으면 비중을 늘리면 보상
            lstm_position_bonus = self.config.get('lstm_position_bonus', 5.0)
            if self.lstm_model is not None and self.sequences_data is not None and self.current_step < len(self.sequences_data):
                try:
                    sequence = self.sequences_data[self.current_step].reshape(1, -1, 5)
                    price_pred = self.lstm_model.predict(sequence, current_price=current_price)
                    target_price_val = price_pred.get('target_price', current_price * 1.02)
                    stop_loss_val = price_pred.get('stop_loss', current_price * 0.98)
                    target_price = float(np.asarray(target_price_val).flatten()[0]) if hasattr(target_price_val, '__len__') else float(target_price_val)
                    stop_loss = float(np.asarray(stop_loss_val).flatten()[0]) if hasattr(stop_loss_val, '__len__') else float(stop_loss_val)
                    
                    # LSTM 보상 스케일 (설정 파일에서 읽기)
                    lstm_bonus_scale = self.config.get('lstm_position_bonus_scale', 10.0)
                    lstm_penalty_scale = self.config.get('lstm_position_penalty_scale', 5.0)
                    
                    # 현재 가격이 목표가보다 높은 경우 (고평가)
                    if current_price > target_price:
                        # 비중을 줄이면 보상 (변화량이 클수록 보상)
                        if new_position < position_scalar:  # 포지션 감소
                            reward = float(reward + lstm_position_bonus * position_change * lstm_bonus_scale)
                        # 비중을 늘리면 페널티
                        elif new_position > position_scalar:  # 포지션 증가
                            reward = float(reward - lstm_position_bonus * position_change * lstm_penalty_scale)
                    
                    # 현재 가격이 손절가보다 낮은 경우 (저평가)
                    elif current_price < stop_loss:
                        # 비중을 늘리면 보상 (변화량이 클수록 보상)
                        if new_position > position_scalar:  # 포지션 증가
                            reward = float(reward + lstm_position_bonus * position_change * lstm_bonus_scale)
                        # 비중을 줄이면 페널티
                        elif new_position < position_scalar:  # 포지션 감소
                            reward = float(reward - lstm_position_bonus * position_change * lstm_penalty_scale)
                except:
                    pass
            
            # 포지션 업데이트 (스칼라로 보장)
            if hasattr(new_position, '__len__') and not isinstance(new_position, str):
                self.position = float(np.asarray(new_position).flatten()[0])
            else:
                self.position = float(new_position)
            
            self.current_step += 1
            
            # 종료 조건
            terminated = (self.current_step >= len(self.price_data) - 1)
            truncated = False  # 샘플에서는 truncation 없음
            
            observation = self._get_observation()
            info = {}
            
            return observation, reward, terminated, truncated, info
        
        def _get_observation(self):
            """관측값 생성 (XGBoost/LSTM 정보 포함)"""
            if self.current_step >= len(self.price_data):
                observation_size = self.config.get('observation_space_size', 14)
                return np.zeros(observation_size)
            
            # 가격을 스칼라로 안전하게 변환
            current_price_val = self.price_data[self.current_step]
            if hasattr(current_price_val, '__len__') and not isinstance(current_price_val, str):
                current_price = float(np.asarray(current_price_val).flatten()[0])
            else:
                current_price = float(current_price_val)
            
            # 가격 변화율
            price_change = 0.0
            if self.current_step > 0:
                prev_price_val = self.price_data[self.current_step - 1]
                if hasattr(prev_price_val, '__len__') and not isinstance(prev_price_val, str):
                    prev_price = float(np.asarray(prev_price_val).flatten()[0])
                else:
                    prev_price = float(prev_price_val)
                price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
            
            # 단기/장기 이동평균 (간단한 추세 지표)
            ma_short = current_price
            ma_long = current_price
            if self.current_step >= 5:
                ma_short_val = np.mean(self.price_data[max(0, self.current_step-5):self.current_step+1])
                if hasattr(ma_short_val, '__len__') and not isinstance(ma_short_val, str):
                    ma_short = float(np.asarray(ma_short_val).flatten()[0])
                else:
                    ma_short = float(ma_short_val)
            if self.current_step >= 20:
                ma_long_val = np.mean(self.price_data[max(0, self.current_step-20):self.current_step+1])
                if hasattr(ma_long_val, '__len__') and not isinstance(ma_long_val, str):
                    ma_long = float(np.asarray(ma_long_val).flatten()[0])
                else:
                    ma_long = float(ma_long_val)
            
            # 변동성 (최근 10일)
            volatility = 0.01
            if self.current_step >= 10:
                recent_prices = self.price_data[max(0, self.current_step-10):self.current_step+1]
                if len(recent_prices) > 1:
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    volatility_val = np.std(returns) if len(returns) > 0 else 0.01
                    if hasattr(volatility_val, '__len__') and not isinstance(volatility_val, str):
                        volatility = float(np.asarray(volatility_val).flatten()[0])
                    else:
                        volatility = float(volatility_val)
            
            # 모멘텀 (최근 5일 수익률)
            momentum = 0.0
            if self.current_step >= 5:
                prev_price_5_val = self.price_data[self.current_step - 5]
                if hasattr(prev_price_5_val, '__len__') and not isinstance(prev_price_5_val, str):
                    prev_price_5 = float(np.asarray(prev_price_5_val).flatten()[0])
                else:
                    prev_price_5 = float(prev_price_5_val)
                momentum = (current_price - prev_price_5) / prev_price_5 if prev_price_5 > 0 else 0.0
            
            # XGBoost/LSTM 예측 정보 (학습 시에도 포함)
            buy_prob = 0.5
            sell_prob = 0.5
            target_price_ratio = 1.02
            stop_loss_ratio = 0.98
            price_confidence = 0.5
            
            if self.xgb_model is not None and self.features_data is not None and self.current_step < len(self.features_data):
                try:
                    features = self.features_data.iloc[self.current_step].values.reshape(1, -1)
                    direction_pred = self.xgb_model.predict_proba(features)[0]
                    direction_pred = np.asarray(direction_pred).flatten()  # 1차원 배열로 변환
                    if len(direction_pred) >= 3:
                        # 배열 요소를 스칼라로 안전하게 변환
                        buy_prob_val = direction_pred[1]
                        if hasattr(buy_prob_val, '__len__') and not isinstance(buy_prob_val, str):
                            buy_prob = float(np.asarray(buy_prob_val).flatten()[0])
                        else:
                            buy_prob = float(buy_prob_val)
                        
                        sell_prob_val = direction_pred[2]
                        if hasattr(sell_prob_val, '__len__') and not isinstance(sell_prob_val, str):
                            sell_prob = float(np.asarray(sell_prob_val).flatten()[0])
                        else:
                            sell_prob = float(sell_prob_val)
                except:
                    pass
            
            if self.lstm_model is not None and self.sequences_data is not None and self.current_step < len(self.sequences_data):
                try:
                    sequence = self.sequences_data[self.current_step].reshape(1, -1, 5)  # (1, sequence_length, 5)
                    price_pred = self.lstm_model.predict(sequence, current_price=current_price)
                    # 배열을 스칼라로 안전하게 변환
                    target_price_val = price_pred.get('target_price', current_price * 1.02)
                    stop_loss_val = price_pred.get('stop_loss', current_price * 0.98)
                    target_price = float(np.asarray(target_price_val).flatten()[0]) if hasattr(target_price_val, '__len__') else float(target_price_val)
                    stop_loss = float(np.asarray(stop_loss_val).flatten()[0]) if hasattr(stop_loss_val, '__len__') else float(stop_loss_val)
                    target_price_ratio = target_price / current_price if current_price > 0 else 1.02
                    stop_loss_ratio = stop_loss / current_price if current_price > 0 else 0.98
                    price_confidence_val = price_pred.get('price_confidence', 0.5)
                    price_confidence = float(np.asarray(price_confidence_val).flatten()[0]) if hasattr(price_confidence_val, '__len__') else float(price_confidence_val)
                except:
                    pass
            
            price_normalization = self.config.get('price_normalization', 200)
            observation_size = self.config.get('observation_space_size', 14)  # XGBoost/LSTM 정보 추가로 14차원
            
            # position을 스칼라로 변환
            position_val = self.position
            if hasattr(position_val, '__len__') and not isinstance(position_val, str):
                position_scalar = float(np.asarray(position_val).flatten()[0])
            else:
                position_scalar = float(position_val)
            
            # balance를 스칼라로 변환
            balance_val = self.balance
            if hasattr(balance_val, '__len__') and not isinstance(balance_val, str):
                balance_scalar = float(np.asarray(balance_val).flatten()[0])
            else:
                balance_scalar = float(balance_val)
            
            # 모든 값을 스칼라로 보장
            obs = [
                float(position_scalar),  # [0] 현재 포지션
                float(balance_scalar / self.initial_balance),  # [1] 정규화된 잔액
                float(current_price / price_normalization),  # [2] 정규화된 가격
                float(price_change),  # [3] 가격 변화율
                float((ma_short - ma_long) / current_price if current_price > 0 else 0),  # [4] 이동평균 차이
                float(momentum),  # [5] 모멘텀
                float(volatility),  # [6] 변동성
                float(min(self.current_step / 1000.0, 1.0)),  # [7] 시간 정규화
                float(buy_prob),  # [8] XGBoost BUY 확률
                float(sell_prob),  # [9] XGBoost SELL 확률
                float(target_price_ratio),  # [10] LSTM 목표가 비율
                float(stop_loss_ratio),  # [11] LSTM 손절가 비율
                float(price_confidence),  # [12] LSTM 신뢰도
                float(1.0 if position_scalar != 0.0 else 0.0)  # [13] 포지션 보유 여부
            ]
            
            # observation_space_size에 맞게 조정
            if len(obs) < observation_size:
                obs.extend([0.0] * (observation_size - len(obs)))
            elif len(obs) > observation_size:
                obs = obs[:observation_size]
            
            return np.array(obs, dtype=np.float32)
else:
    # gym이 없을 경우 더미 클래스
    class SimpleTradingEnv:
        def __init__(self, *args, **kwargs):
            self.price_data = args[0] if args else np.array([100])
            self.balance = 100000
            self.position = 0.0
            self.current_step = 0
            self.action_space = None
            self.observation_space = None
        def reset(self):
            return np.zeros(12)
        def step(self, action):
            return np.zeros(12), 0.0, True, {}


class TradingRLAgent:
    """
    샘플: 강화학습 Agent
    XGBoost와 LSTM의 예측을 종합하여 최종 행동을 결정합니다.
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 모델 설정 딕셔너리 (설정 파일에서 로드)
        """
        if config is None:
            config = {
                'learning_rate': 3e-4,
                'total_timesteps': 100000,
                'gamma': 0.99,
                'n_steps': 256,
                'batch_size': 128,
                'n_epochs': 10,
                'ent_coef': 0.05,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'clip_range': 0.2,
                'initial_balance': 100000,
                'position_size': 0.3,
                'observation_space_size': 12,
                'reward_scale': 200,
                'opportunity_cost_penalty': 50,
                'avoidance_bonus': 30,
                'correct_direction_bonus': 100,
                'wrong_direction_penalty': 50,
                'action_bonus': 0.5,
                'transaction_cost': 0.05,
                'price_normalization': 200
            }
        
        self.config = config
        self.agent = None
        self.training_rewards = []  # 학습 중 보상 기록
        self.training_episodes = []  # 에피소드 기록
        print("[샘플] 강화학습 Agent 초기화")
    
    def train(self, price_data: np.ndarray, xgb_model=None, lstm_model=None, 
              features_data=None, sequences_data=None):
        """
        Agent 학습
        
        Args:
            price_data: 가격 데이터 배열
            xgb_model: XGBoost 모델 (학습 시 observation에 포함)
            lstm_model: LSTM 모델 (학습 시 observation에 포함)
            features_data: XGBoost용 피처 데이터 (DataFrame)
            sequences_data: LSTM용 시퀀스 데이터 (numpy array)
        """
        if not STABLE_BASELINES3_AVAILABLE or not GYM_AVAILABLE:
            print("[샘플] stable-baselines3 또는 gym이 설치되지 않아 강화학습을 건너뜁니다.")
            self.agent = None
            return
        
        try:
            print(f"[샘플] 강화학습 학습 시작: {len(price_data)}개 데이터 포인트")
            if xgb_model is not None:
                print("[샘플] XGBoost 모델 정보를 학습에 포함합니다.")
            if lstm_model is not None:
                print("[샘플] LSTM 모델 정보를 학습에 포함합니다.")
            
            # 환경 생성 (XGBoost/LSTM 모델 포함)
            initial_balance = self.config.get('initial_balance', 100000)
            env = SimpleTradingEnv(
                price_data, 
                initial_balance=initial_balance, 
                config=self.config,
                xgb_model=xgb_model,
                lstm_model=lstm_model,
                features_data=features_data,
                sequences_data=sequences_data
            )
            
            # RL Agent 생성 (알고리즘 선택 가능)
            algorithm_name = self.config.get('algorithm', 'PPO').upper()
            if algorithm_name not in RL_ALGORITHMS:
                print(f"[샘플] 경고: 지원하지 않는 알고리즘 '{algorithm_name}', PPO 사용")
                algorithm_name = 'PPO'
            
            AlgorithmClass = RL_ALGORITHMS[algorithm_name]
            print(f"[샘플] {algorithm_name} 알고리즘 사용")
            
            # learning_rate를 float로 변환 (YAML에서 문자열로 읽힐 수 있음)
            lr = self.config.get('learning_rate', 3e-4)
            if isinstance(lr, str):
                lr = float(lr)
            
            # 공통 파라미터
            common_params = {
                'policy': 'MlpPolicy',
                'env': env,
                'learning_rate': lr,
                'gamma': self.config.get('gamma', 0.99),
                'verbose': 1
            }
            
            # 알고리즘별 파라미터 설정
            if algorithm_name in ['PPO', 'A2C']:
                # On-policy 알고리즘 (PPO, A2C)
                algorithm_params = {
                    'n_steps': self.config.get('n_steps', 256),
                    'ent_coef': self.config.get('ent_coef', 0.05),
                    'vf_coef': self.config.get('vf_coef', 0.5),
                    'max_grad_norm': self.config.get('max_grad_norm', 0.5),
                }
                if algorithm_name == 'PPO':
                    algorithm_params['batch_size'] = self.config.get('batch_size', 128)
                    algorithm_params['n_epochs'] = self.config.get('n_epochs', 10)
                    algorithm_params['clip_range'] = self.config.get('clip_range', 0.2)
                elif algorithm_name == 'A2C':
                    algorithm_params['gae_lambda'] = self.config.get('gae_lambda', 0.95)
                    algorithm_params['use_sde'] = self.config.get('use_sde', False)
                    if algorithm_params['use_sde']:
                        algorithm_params['sde_sample_freq'] = self.config.get('sde_sample_freq', -1)
            elif algorithm_name in ['DQN']:
                # DQN (Discrete action space는 현재 환경과 맞지 않지만 지원)
                algorithm_params = {
                    'buffer_size': self.config.get('buffer_size', 100000),
                    'learning_starts': self.config.get('learning_starts', 1000),
                    'batch_size': self.config.get('batch_size', 32),
                    'tau': self.config.get('tau', 1.0),
                    'gamma': self.config.get('gamma', 0.99),
                    'train_freq': self.config.get('dqn_train_freq', 4),
                    'gradient_steps': self.config.get('gradient_steps', 1),
                    'target_update_interval': self.config.get('dqn_target_update_interval', 1000),
                }
            elif algorithm_name in ['SAC', 'TD3', 'DDPG']:
                # Off-policy 알고리즘 (SAC, TD3, DDPG)
                algorithm_params = {
                    'buffer_size': self.config.get('buffer_size', 100000),
                    'learning_starts': self.config.get('learning_starts', 1000),
                    'batch_size': self.config.get('batch_size', 256),
                    'tau': self.config.get('tau', 0.005),
                    'gamma': self.config.get('gamma', 0.99),
                    'train_freq': self.config.get('train_freq', 1),
                    'gradient_steps': self.config.get('gradient_steps', 1),
                }
                if algorithm_name == 'SAC':
                    algorithm_params['ent_coef'] = self.config.get('ent_coef', 'auto')
                    algorithm_params['target_update_interval'] = self.config.get('sac_target_update_interval', 1)
                elif algorithm_name in ['TD3', 'DDPG']:
                    algorithm_params['policy_delay'] = self.config.get('policy_delay', 2) if algorithm_name == 'TD3' else None
                    algorithm_params['target_policy_noise'] = self.config.get('target_policy_noise', 0.2) if algorithm_name == 'TD3' else None
                    algorithm_params['target_noise_clip'] = self.config.get('target_noise_clip', 0.5) if algorithm_name == 'TD3' else None
            
            # 모든 파라미터 병합
            all_params = {**common_params, **algorithm_params}
            # None 값 제거
            all_params = {k: v for k, v in all_params.items() if v is not None}
            
            self.agent = AlgorithmClass(**all_params)
            
            # 조기 종료 콜백 설정 (모든 알고리즘에 적용)
            callbacks = None
            enable_early_stopping = self.config.get('enable_early_stopping', False)
            early_stopping_patience = self.config.get('early_stopping_patience', 10)
            eval_freq = self.config.get('eval_freq', 5000)
            
            if enable_early_stopping and STABLE_BASELINES3_AVAILABLE:
                try:
                    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
                    
                    # 검증 환경 생성 (학습 데이터의 일부를 검증용으로 사용)
                    # 간단하게 동일한 환경을 검증 환경으로 사용 (실제로는 별도 데이터가 필요)
                    val_env = env  # 실제로는 별도 검증 데이터로 환경을 만들어야 함
                    
                    # 조기 종료 조건: patience회 평가 동안 개선이 없으면 종료
                    stop_callback = StopTrainingOnNoModelImprovement(
                        max_no_improvement_evals=early_stopping_patience,
                        min_evals=5,  # 최소 5회는 평가
                        verbose=1
                    )
                    
                    # 평가 콜백: 주기적으로 검증 환경에서 평가
                    eval_callback = EvalCallback(
                        val_env,
                        best_model_save_path=None,  # 모델 저장은 별도로 처리
                        log_path=None,
                        eval_freq=eval_freq,
                        deterministic=True,
                        render=False,
                        callback_on_new_best=stop_callback,
                        verbose=1
                    )
                    callbacks = [eval_callback]
                    print(f"[샘플] 조기 종료 활성화: {early_stopping_patience}회 평가 동안 개선 없으면 종료 (eval_freq: {eval_freq})")
                except Exception as e:
                    print(f"[샘플] 조기 종료 콜백 설정 실패: {e}, 조기 종료 없이 학습 진행")
                    callbacks = None
            
            # 학습 (프로그레스바 활성화, 콜백 포함)
            self.agent.learn(
                total_timesteps=self.config['total_timesteps'], 
                progress_bar=True,
                callback=callbacks
            )
            
            # 학습 후 평가를 통해 보상 곡선 생성
            self._collect_training_metrics(env)
            
            print("[샘플] 강화학습 학습 완료")
        except Exception as e:
            print(f"[샘플] 강화학습 학습 중 오류 발생: {e}")
            print("[샘플] 강화학습은 건너뛰고 기본 행동을 사용합니다.")
            self.agent = None
    
    def predict(self, state: Dict) -> Dict:
        """
        행동 예측
        
        Args:
            state: 상태 딕셔너리 (direction_signal, price_target, current_state 포함)
        
        Returns:
            행동 딕셔너리
        """
        if self.agent is None:
            # 학습되지 않은 경우 기본 행동 반환
            return {
                'action_type': 0,  # HOLD
                'position_size': 0.0,
                'confidence': 0.5
            }
        
        # 상태를 관측 벡터로 변환
        obs = self._state_to_observation(state)
        
        # 예측 (Box action: [position_size])
        # deterministic=False로 변경하여 탐험 허용 (학습된 정책의 다양성 활용)
        action, _ = self.agent.predict(obs, deterministic=False)
        
        # Action: [position_size] - 주식은 position_size만 사용
        # position_size: 0.0 (청산) ~ 1.0 (최대 롱)
        # action을 1차원 배열로 변환하고 스칼라로 추출
        action = np.asarray(action).flatten()
        action = np.clip(action, 0.0, 1.0)
        
        # position_size 추출 및 이산화 (10% 단위로만 선택 가능)
        if len(action) > 0:
            position_size_val = action[0]
            if hasattr(position_size_val, '__len__') and not isinstance(position_size_val, str):
                position_size = float(np.asarray(position_size_val).flatten()[0])
            else:
                position_size = float(position_size_val)
        else:
            position_size = 0.0
        
        position_size = float(np.clip(position_size, 0.0, 1.0))
        
        # 액션 이산화: 10% 단위로만 선택 가능 (0.0, 0.1, 0.2, ..., 1.0)
        # 학습 환경과 동일하게 이산화하여 일관성 유지
        min_meaningful_change = self.config.get('min_meaningful_position_change', 0.10)
        position_size_before_discretize = position_size
        position_size = round(position_size / min_meaningful_change) * min_meaningful_change
        position_size = float(np.clip(position_size, 0.0, 1.0))
        
        # action_type 제거: position_size만 반환
        return {
            'position_size': position_size,  # RL이 학습한 목표 포지션 크기
            'confidence': 0.7  # 샘플이므로 고정값
        }
    
    def _state_to_observation(self, state: Dict) -> np.ndarray:
        """상태를 관측 벡터로 변환 (14차원, 학습 환경과 동일)"""
        direction = state.get('direction', {})
        price_target = state.get('price_target', {})
        current_state = state.get('current_state', {})
        
        # 값들을 스칼라로 안전하게 변환
        current_price_val = current_state.get('current_price', 100.0)
        if hasattr(current_price_val, '__len__') and not isinstance(current_price_val, str):
            current_price = float(np.asarray(current_price_val).flatten()[0])
        else:
            current_price = float(current_price_val)
        
        buy_prob_val = direction.get('buy_prob', 0.5)
        if hasattr(buy_prob_val, '__len__') and not isinstance(buy_prob_val, str):
            buy_prob = float(np.asarray(buy_prob_val).flatten()[0])
        else:
            buy_prob = float(buy_prob_val)
        
        sell_prob_val = direction.get('sell_prob', 0.5)
        if hasattr(sell_prob_val, '__len__') and not isinstance(sell_prob_val, str):
            sell_prob = float(np.asarray(sell_prob_val).flatten()[0])
        else:
            sell_prob = float(sell_prob_val)
        
        target_price_val = price_target.get('target_price', current_price * 1.02)
        if hasattr(target_price_val, '__len__') and not isinstance(target_price_val, str):
            target_price = float(np.asarray(target_price_val).flatten()[0])
        else:
            target_price = float(target_price_val)
        
        stop_loss_val = price_target.get('stop_loss', current_price * 0.98)
        if hasattr(stop_loss_val, '__len__') and not isinstance(stop_loss_val, str):
            stop_loss = float(np.asarray(stop_loss_val).flatten()[0])
        else:
            stop_loss = float(stop_loss_val)
        
        price_confidence_val = price_target.get('price_confidence', 0.5)
        if hasattr(price_confidence_val, '__len__') and not isinstance(price_confidence_val, str):
            price_confidence = float(np.asarray(price_confidence_val).flatten()[0])
        else:
            price_confidence = float(price_confidence_val)
        
        # 가격 변화율 (간단히 0으로 설정, 실제로는 이전 가격 필요)
        price_change = 0.0
        
        # 이동평균 차이 (간단히 0으로 설정, 실제로는 이전 가격들 필요)
        ma_diff = 0.0
        
        # 모멘텀 (간단히 0으로 설정)
        momentum = 0.0
        
        # 변동성 (스칼라로 안전하게 변환)
        volatility_val = current_state.get('market_volatility', 0.01)
        if hasattr(volatility_val, '__len__') and not isinstance(volatility_val, str):
            volatility = float(np.asarray(volatility_val).flatten()[0])
        else:
            volatility = float(volatility_val)
        
        # 목표가/손절가 비율
        target_price_ratio = target_price / current_price if current_price > 0 else 1.02
        stop_loss_ratio = stop_loss / current_price if current_price > 0 else 0.98
        
        price_normalization = self.config.get('price_normalization', 200)
        observation_size = self.config.get('observation_space_size', 14)  # 학습 환경과 동일
        initial_balance = self.config.get('initial_balance', 100000)
        
        # current_position과 balance를 스칼라로 변환
        current_position_val = current_state.get('current_position', 0.0)
        if hasattr(current_position_val, '__len__') and not isinstance(current_position_val, str):
            current_position = float(np.asarray(current_position_val).flatten()[0])
        else:
            current_position = float(current_position_val)
        
        balance_val = current_state.get('balance', initial_balance)
        if hasattr(balance_val, '__len__') and not isinstance(balance_val, str):
            balance = float(np.asarray(balance_val).flatten()[0])
        else:
            balance = float(balance_val)
        
        # 모든 값을 스칼라로 보장
        obs = [
            float(current_position),  # [0] 현재 포지션
            float(balance / initial_balance),  # [1] 정규화된 잔액
            float(current_price / price_normalization),  # [2] 정규화된 가격
            float(price_change),  # [3] 가격 변화율
            float(ma_diff),  # [4] 이동평균 차이
            float(momentum),  # [5] 모멘텀
            float(volatility),  # [6] 변동성
            float(0.5),  # [7] 시간 정규화 (샘플이므로 고정값)
            float(buy_prob),  # [8] XGBoost BUY 확률
            float(sell_prob),  # [9] XGBoost SELL 확률
            float(target_price_ratio),  # [10] LSTM 목표가 비율
            float(stop_loss_ratio),  # [11] LSTM 손절가 비율
            float(price_confidence),  # [12] LSTM 신뢰도
            float(1.0 if current_position != 0.0 else 0.0)  # [13] 포지션 보유 여부
        ]
        
        # observation_space_size에 맞게 조정
        if len(obs) < observation_size:
            obs.extend([0.0] * (observation_size - len(obs)))
        elif len(obs) > observation_size:
            obs = obs[:observation_size]
        
        return np.array(obs, dtype=np.float32)
    
    def save(self, filepath: str):
        """모델 저장"""
        if self.agent is None:
            raise ValueError("[샘플] 저장할 모델이 없습니다.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)
        print(f"[샘플] 모델 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        try:
            # 환경이 필요하므로 임시 환경 생성
            initial_balance = self.config.get('initial_balance', 100000)
            temp_env = SimpleTradingEnv(np.array([100, 101, 102]), initial_balance=initial_balance, config=self.config)
            
            # 알고리즘 확인 (파일명이나 메타데이터에서 추출 가능하지만, 일단 config에서 읽기)
            algorithm_name = self.config.get('algorithm', 'PPO').upper()
            if algorithm_name not in RL_ALGORITHMS:
                algorithm_name = 'PPO'
            
            AlgorithmClass = RL_ALGORITHMS[algorithm_name]
            self.agent = AlgorithmClass.load(filepath, env=temp_env)
            print(f"[샘플] 모델 로드 완료: {filepath} ({algorithm_name})")
        except Exception as e:
            print(f"[샘플] 모델 로드 실패: {e}")
            self.agent = None
    
    def _collect_training_metrics(self, env):
        """학습 후 평가를 통해 메트릭 수집"""
        if self.agent is None:
            return
        
        # 여러 에피소드 평가
        episode_rewards = []
        episode_lengths = []
        episode_actions = []
        
        for _ in range(10):  # 10개 에피소드 평가
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            actions_in_episode = []
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # reward를 스칼라로 변환
                if hasattr(reward, '__len__') and not isinstance(reward, str):
                    reward_scalar = float(np.asarray(reward).flatten()[0])
                else:
                    reward_scalar = float(reward)
                episode_reward = float(episode_reward + reward_scalar)
                episode_length += 1
                # action을 스칼라로 변환 후 int로 변환
                if hasattr(action, '__len__') and not isinstance(action, str):
                    action_scalar = float(np.asarray(action).flatten()[0])
                else:
                    action_scalar = float(action)
                actions_in_episode.append(int(action_scalar))
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_actions.extend(actions_in_episode)
        
        self.training_rewards = episode_rewards
        self.training_episodes = list(range(len(episode_rewards)))
        self.training_actions = episode_actions
    
    def visualize_results(self, price_data: np.ndarray, save_dir: Optional[str] = None):
        """
        학습 결과 시각화
        
        Args:
            price_data: 가격 데이터 (평가용)
            save_dir: 이미지 저장 디렉토리 (선택사항)
        """
        if self.agent is None:
            print("[샘플] 학습된 RL Agent가 없어 시각화를 건너뜁니다.")
            return
        
        try:
            # 한글 폰트 설정
            import platform
            system = platform.system()
            if system == 'Windows':
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif system == 'Darwin':  # macOS
                plt.rcParams['font.family'] = 'AppleGothic'
            else:  # Linux
                try:
                    plt.rcParams['font.family'] = 'NanumGothic'
                except:
                    plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        print("\n[샘플] 강화학습 학습 결과 시각화 중...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. 에피소드별 보상 곡선
        if self.training_rewards:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.training_episodes, self.training_rewards, 'o-', label='에피소드 보상', color='blue', alpha=0.7)
            if len(self.training_rewards) > 1:
                # 이동평균
                window = min(5, len(self.training_rewards) // 2)
                if window > 1:
                    moving_avg = np.convolve(self.training_rewards, np.ones(window)/window, mode='valid')
                    ax.plot(self.training_episodes[window-1:], moving_avg, '--', label=f'이동평균 ({window})', color='red', linewidth=2)
            ax.set_xlabel('에피소드')
            ax.set_ylabel('누적 보상')
            ax.set_title('강화학습 에피소드별 보상 곡선')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_episode_rewards.png'), dpi=150)
                print(f"[샘플] 에피소드 보상 곡선 저장: {os.path.join(save_dir, 'rl_episode_rewards.png')}")
            plt.close()
        
        # 2. 평가 실행 (가격 데이터로)
        initial_balance = self.config.get('initial_balance', 100000)
        env = SimpleTradingEnv(price_data, initial_balance=initial_balance, config=self.config)
        obs, _ = env.reset()
        
        rewards = []
        actions = []
        positions = []
        prices = []
        done = False
        step_count = 0
        
        while not done and step_count < len(price_data) - 1:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # reward를 스칼라로 변환
            if hasattr(reward, '__len__') and not isinstance(reward, str):
                reward_scalar = float(np.asarray(reward).flatten()[0])
            else:
                reward_scalar = float(reward)
            rewards.append(reward_scalar)
            
            # action을 스칼라로 변환 후 int로 변환
            if hasattr(action, '__len__') and not isinstance(action, str):
                action_scalar = float(np.asarray(action).flatten()[0])
            else:
                action_scalar = float(action)
            actions.append(int(action_scalar))
            
            # env.position을 스칼라로 변환
            position_val = env.position
            if hasattr(position_val, '__len__') and not isinstance(position_val, str):
                position_scalar = float(np.asarray(position_val).flatten()[0])
            else:
                position_scalar = float(position_val)
            positions.append(position_scalar)
            
            # price를 스칼라로 변환
            price_val = price_data[min(env.current_step, len(price_data)-1)]
            if hasattr(price_val, '__len__') and not isinstance(price_val, str):
                price_scalar = float(np.asarray(price_val).flatten()[0])
            else:
                price_scalar = float(price_val)
            prices.append(price_scalar)
            
            step_count += 1
        
        # 3. 누적 보상 곡선
        if rewards:
            fig, ax = plt.subplots(figsize=(12, 6))
            cum_rewards = np.cumsum(rewards)
            ax.plot(cum_rewards, label='누적 보상', color='tab:blue', linewidth=2)
            if len(rewards) > 20:
                mov_avg = np.convolve(rewards, np.ones(20)/20, mode='valid')
                ax.plot(range(19, 19+len(mov_avg)), mov_avg, label='보상 이동평균 (20)', color='tab:orange', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('스텝')
            ax.set_ylabel('보상')
            ax.set_title('강화학습 평가: 누적 보상 및 이동평균')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_cumulative_rewards.png'), dpi=150)
                print(f"[샘플] 누적 보상 곡선 저장: {os.path.join(save_dir, 'rl_cumulative_rewards.png')}")
            plt.close()
        
        # 4. 액션 분포
        if actions:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 액션 타임라인
            axes[0].step(range(len(actions)), actions, where='post', color='tab:purple', linewidth=1)
            axes[0].set_yticks([0, 1, 2])
            axes[0].set_yticklabels(['HOLD', 'BUY', 'SELL'])
            axes[0].set_xlabel('스텝')
            axes[0].set_ylabel('액션')
            axes[0].set_title('액션 타임라인')
            axes[0].grid(True, alpha=0.3)
            
            # 액션 분포 (히스토그램)
            action_counts = [actions.count(0), actions.count(1), actions.count(2)]
            axes[1].bar(['HOLD', 'BUY', 'SELL'], action_counts, color=['gray', 'green', 'red'], alpha=0.7)
            axes[1].set_ylabel('빈도')
            axes[1].set_title('액션 분포')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_action_distribution.png'), dpi=150)
                print(f"[샘플] 액션 분포 저장: {os.path.join(save_dir, 'rl_action_distribution.png')}")
            plt.close()
        
        # 5. 가격과 포지션 변화
        if prices and positions:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # 가격 차트
            axes[0].plot(prices, color='tab:gray', label='가격', linewidth=1.5)
            # BUY/SELL 마커
            buy_indices = [i for i, a in enumerate(actions) if a == 1]
            sell_indices = [i for i, a in enumerate(actions) if a == 2]
            if buy_indices:
                axes[0].scatter(buy_indices, [prices[i] for i in buy_indices], 
                             color='tab:green', s=30, marker='^', label='BUY', zorder=5)
            if sell_indices:
                axes[0].scatter(sell_indices, [prices[i] for i in sell_indices], 
                             color='tab:red', s=30, marker='v', label='SELL', zorder=5)
            axes[0].set_ylabel('가격')
            axes[0].set_title('가격 및 거래 신호')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 포지션 변화
            axes[1].plot(positions, color='tab:brown', linewidth=1.5)
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[1].set_xlabel('스텝')
            axes[1].set_ylabel('포지션')
            axes[1].set_title('포지션 변화')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_price_position.png'), dpi=150)
                print(f"[샘플] 가격 및 포지션 차트 저장: {os.path.join(save_dir, 'rl_price_position.png')}")
            plt.close()
        
        print("[샘플] 강화학습 시각화 완료")
