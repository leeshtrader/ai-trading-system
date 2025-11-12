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
        buy_threshold = self.ensemble_config.get('buy_threshold', 0.35)
        sell_threshold = self.ensemble_config.get('sell_threshold', 0.35)
        strong_signal_threshold = self.ensemble_config.get('strong_signal_threshold', 0.4)
        base_position_size_multiplier = self.ensemble_config.get('base_position_size_multiplier', 0.4)
        max_position_size = self.ensemble_config.get('max_position_size', 0.4)
        rl_match_position_multiplier = self.ensemble_config.get('rl_match_position_multiplier', 1.2)
        max_position_size_rl_match = self.ensemble_config.get('max_position_size_rl_match', 0.5)
        default_target_ratio = self.config.get('labeling', {}).get('default_target_ratio', 1.02)
        default_stop_loss_ratio = self.config.get('labeling', {}).get('default_stop_loss_ratio', 0.98)
        
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
        
        # RL Agent의 결정
        rl_action_type = rl_action['action_type']
        
        # RL Agent가 HOLD를 반환하더라도, XGBoost/LSTM의 신호가 강하면 거래 실행
        # RL Agent는 "필터" 역할: 신호가 약하면 HOLD, 강하면 거래 허용
        if rl_action_type == 0:  # RL Agent가 HOLD
            # XGBoost/LSTM의 신호 강도 확인
            if buy_prob > buy_threshold and buy_prob > sell_prob and buy_prob > hold_prob:
                action_type = 1  # BUY
                position_size = min(buy_prob * base_position_size_multiplier, max_position_size)
            elif sell_prob > sell_threshold and sell_prob > buy_prob and sell_prob > hold_prob:
                action_type = 2  # SELL
                position_size = min(sell_prob * base_position_size_multiplier, max_position_size)
            else:
                action_type = 0  # HOLD
                position_size = 0.0
        else:
            # RL Agent가 BUY/SELL을 반환한 경우
            # RL Agent의 결정을 우선하되, XGBoost/LSTM 신호와 일치하면 더 큰 포지션
            action_type = rl_action_type
            base_size = rl_action['position_size']
            
            # XGBoost/LSTM 신호와 일치하면 포지션 크기 증가
            if action_type == 1 and buy_prob > strong_signal_threshold:  # BUY 신호 일치
                position_size = min(base_size * rl_match_position_multiplier, max_position_size_rl_match)
            elif action_type == 2 and sell_prob > strong_signal_threshold:  # SELL 신호 일치
                position_size = min(base_size * rl_match_position_multiplier, max_position_size_rl_match)
            else:
                position_size = base_size
        
        target_price = price_pred.get('target_price', current_state.get('current_price', 0) * default_target_ratio)
        stop_loss = price_pred.get('stop_loss', current_state.get('current_price', 0) * default_stop_loss_ratio)
        confidence = self._calculate_confidence(direction_pred, price_pred, rl_action)
        
        # 4. 최종 신호 구성
        signal = {
            'action': self._action_type_to_string(action_type),
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
    
    def _action_type_to_string(self, action_type: int) -> str:
        """행동 타입을 문자열로 변환"""
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_map.get(action_type, 'HOLD')
    
    def _calculate_confidence(self, direction: Dict, price: Dict, rl_action: Dict) -> float:
        """신뢰도 계산"""
        direction_conf = max(direction.get('buy_prob', 0), direction.get('sell_prob', 0))
        price_conf = price.get('price_confidence', 0.5)
        rl_conf = rl_action.get('confidence', 0.5)
        
        # 설정 파일에서 가중치 읽기
        direction_weight = self.ensemble_config.get('direction_weight', 0.3)
        price_weight = self.ensemble_config.get('price_weight', 0.4)
        rl_weight = self.ensemble_config.get('rl_weight', 0.3)
        
        # 가중 평균
        return (
            direction_conf * direction_weight +
            price_conf * price_weight +
            rl_conf * rl_weight
        )

