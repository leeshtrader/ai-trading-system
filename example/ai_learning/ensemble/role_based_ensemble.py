"""
샘플: 역할 분담 앙상블
역할 분담 앙상블 샘플 구현

주의: 본 모듈은 샘플/데모 목적입니다.

Sample: Role-Based Ensemble
Sample implementation of role-based ensemble for trading signals.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import numpy as np
from typing import Dict
import sys
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models')
sys.path.insert(0, models_dir)

from xgboost_direction import XGBoostDirectionModel
from lstm_price import LSTMPriceModel
from rl_agent import TradingRLAgent


class RoleBasedEnsemble:
    """
    샘플: 역할 분담 앙상블
    XGBoost, LSTM, 강화학습의 예측을 통합하여 최종 거래 신호를 생성합니다.
    """
    
    def __init__(self, xgb_model: XGBoostDirectionModel, 
                 lstm_model: LSTMPriceModel, 
                 rl_agent: TradingRLAgent,
                 config: Dict = None):
        """
        Args:
            xgb_model: XGBoost 방향 예측 모델
            lstm_model: LSTM 가격 예측 모델
            rl_agent: 강화학습 Agent
            config: 설정 딕셔너리
        """
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.rl = rl_agent
        self.config = config or {}
        self.ensemble_config = self.config.get('ensemble', {})
        print("[샘플] 역할 분담 앙상블 초기화")
    
    def generate_signal(self, features: np.ndarray, sequences: np.ndarray, 
                       current_state: Dict) -> Dict:
        """
        최종 거래 신호 생성
        
        Args:
            features: XGBoost용 구조화된 피처
            sequences: LSTM용 시계열 데이터
            current_state: 현재 상태 정보
        
        Returns:
            거래 신호 딕셔너리
        """
        # 1. XGBoost 방향 예측
        direction_pred = self.xgb.predict(features)
        
        # 설정 파일에서 값 읽기
        default_current_price = self.ensemble_config.get('default_current_price', 150.0)
        # buy_threshold, sell_threshold는 XGBoost 신호 분석용 (포지션 크기 계산에는 사용 안 함)
        buy_threshold = self.ensemble_config.get('buy_threshold', 0.35)
        sell_threshold = self.ensemble_config.get('sell_threshold', 0.35)
        strong_signal_threshold = self.ensemble_config.get('strong_signal_threshold', 0.4)
        
        # 2. LSTM 가격 예측 (현재 가격 전달)
        current_price = current_state.get('current_price', default_current_price)
        price_pred = self.lstm.predict(sequences, current_price=current_price)
        
        # 3. 강화학습 최종 결정
        state = {
            'direction': direction_pred,
            'price_target': price_pred,
            'current_state': current_state
        }
        
        rl_action = self.rl.predict(state)
        
        # XGBoost/LSTM 정보 추출
        buy_prob = direction_pred.get('buy_prob', 0.0)
        sell_prob = direction_pred.get('sell_prob', 0.0)
        hold_prob = direction_pred.get('hold_prob', 0.0)
        price_confidence = price_pred.get('price_confidence', 0.5)
        
        # RL Agent의 최종 결정: position_size만 사용
        # XGBoost는 확률만 제공 (RL의 입력으로 사용), LSTM은 목표가/손절가만 제공
        # RL이 목표 포지션 크기(position_size)를 결정
        position_size = rl_action.get('position_size', 0.0)
        
        # 목표가/손절가 계산 (변동성 기반 또는 LSTM 예측값 사용)
        labeling_config = self.config.get('labeling', {})
        
        # 변동성 기반 계산
        market_volatility = current_state.get('market_volatility', 0.01)
        volatility_multiplier = labeling_config.get('volatility_multiplier', 2.0)
        consider_transaction_cost = labeling_config.get('consider_transaction_cost', True)
        
        # 수수료 고려한 최소/최대 기준 계산
        if consider_transaction_cost:
            # 백테스트 설정에서 수수료율 읽기
            backtest_config = self.config.get('backtest', {})
            transaction_cost_rate = backtest_config.get('transaction_cost_rate', 0.001)  # 0.1%
            total_transaction_cost = transaction_cost_rate * 2  # 매수 + 매도 = 2배
            
            # 최소 익절: 수수료 + 최소 순수익(1.3%)
            min_target_ratio_base = labeling_config.get('min_target_ratio', 1.015)
            min_target_ratio = max(min_target_ratio_base, 1.0 + total_transaction_cost + 0.013)
            
            # 최대 손절: 수수료 + 최소 손실 허용(1.3%)
            max_stop_loss_ratio_base = labeling_config.get('max_stop_loss_ratio', 0.985)
            max_stop_loss_ratio = min(max_stop_loss_ratio_base, 1.0 - total_transaction_cost - 0.013)
        else:
            min_target_ratio = labeling_config.get('min_target_ratio', 1.015)
            max_stop_loss_ratio = labeling_config.get('max_stop_loss_ratio', 0.985)
        
        max_target_ratio = labeling_config.get('max_target_ratio', 1.15)
        min_stop_loss_ratio = labeling_config.get('min_stop_loss_ratio', 0.85)
        
        # 변동성 기반 비율 계산 (2시그마 등)
        if market_volatility > 0:
            target_ratio = 1.0 + (market_volatility * volatility_multiplier)
            stop_loss_ratio = 1.0 - (market_volatility * volatility_multiplier)
            # 최소/최대 비율로 제한
            target_ratio = np.clip(target_ratio, min_target_ratio, max_target_ratio)
            stop_loss_ratio = np.clip(stop_loss_ratio, min_stop_loss_ratio, max_stop_loss_ratio)
        else:
            # 변동성이 없으면 최소/최대 범위의 중간값 사용
            target_ratio = (min_target_ratio + max_target_ratio) / 2.0
            stop_loss_ratio = (min_stop_loss_ratio + max_stop_loss_ratio) / 2.0
        
        # LSTM 예측값이 있으면 우선 사용, 없으면 변동성 기반 값 사용
        target_price = price_pred.get('target_price', current_price * target_ratio)
        stop_loss = price_pred.get('stop_loss', current_price * stop_loss_ratio)
        
        # 신뢰도: RL의 confidence만 사용 (XGBoost/LSTM의 실제 신뢰도는 RL의 observation으로 이미 포함됨)
        confidence = rl_action.get('confidence', 0.5)
        
        # 4. 최종 신호 구성
        signal = {
            'action': 'BUY' if position_size > 0.0 else 'HOLD',  # 로깅용 (실제로는 position_size만 사용)
            'size': position_size,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'confidence': confidence,
            'reasoning': {
                'direction_signal': direction_pred,
                'price_prediction': price_pred,
                'rl_decision': rl_action
            }
        }
        
        return signal

