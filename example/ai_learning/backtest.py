"""
샘플: 백테스트 모듈
과거 데이터로 학습된 모델의 성과를 검증합니다.

주의: 본 모듈은 샘플/데모 목적입니다.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import sys
import os

# 경로 설정 - 같은 디렉토리의 모듈 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models.xgboost_direction import XGBoostDirectionModel
from models.lstm_price import LSTMPriceModel
from models.rl_agent import TradingRLAgent
from ensemble.role_based_ensemble import RoleBasedEnsemble

# 상위 디렉토리로 이동하여 feature_engineering import
example_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, example_dir)
from feature_engineering.feature_engineering import FeatureEngineering


class Backtester:
    """
    샘플: 백테스트 엔진
    과거 데이터로 거래 시뮬레이션을 수행합니다.
    """
    
    def __init__(self, initial_balance: float = 100000, config: Dict = None):
        """
        Args:
            initial_balance: 초기 자본금
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # 현재 포지션 (-1.0 ~ 1.0)
        self.entry_price = 0.0  # 진입가
        self.entry_timestamp = None  # 진입 시점
        # 익절/손절 제거: RL의 포지션 관리(비중)로만 리스크 관리
        self.entry_meta = {}  # 진입 시점 메타데이터 (XGB/LSTM/RL)
        self.trade_history = []
        self.signal_history = []  # 신호 기록 (모델별 기여도 분석용)
        print(f"[샘플] 백테스트 초기화: 초기 자본 {initial_balance:,.0f}")
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    xgb_model: XGBoostDirectionModel,
                    lstm_model: LSTMPriceModel,
                    rl_agent: TradingRLAgent,
                    feature_eng: FeatureEngineering,
                    start_idx: int = None,
                    end_idx: int = None) -> Dict:
        """
        백테스트 실행
        
        Args:
            data: OHLCV 데이터 (기술적 지표 포함)
            xgb_model: 학습된 XGBoost 모델
            lstm_model: 학습된 LSTM 모델
            rl_agent: 학습된 강화학습 Agent
            feature_eng: 피처 엔지니어링 모듈
            start_idx: 시작 인덱스 (None이면 처음부터)
            end_idx: 종료 인덱스 (None이면 끝까지)
        
        Returns:
            백테스트 결과 딕셔너리
        """
        print("\n[샘플] 백테스트 시작")
        
        # 인덱스 설정
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(data)
        
        # 초기화
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        # 익절/손절 제거: RL의 포지션 관리(비중)로만 리스크 관리
        self.entry_meta = {}
        self.trade_history = []
        self.signal_history = []
        
        # 피처 준비
        features_df = feature_eng.prepare_features_for_xgboost(data)
        lstm_config = self.config.get('models', {}).get('lstm', {})
        sequence_length = lstm_config.get('sequence_length', 20)
        sequences = feature_eng.prepare_sequences_for_lstm(data, sequence_length=sequence_length)
        
        # 앙상블 생성
        ensemble = RoleBasedEnsemble(xgb_model, lstm_model, rl_agent, self.config)
        
        # 백테스트 실행 (시퀀스가 필요한 만큼 건너뛰기)
        sequence_start = self.config.get('backtest', {}).get('sequence_start_offset', 20)
        
        for i in range(max(start_idx, sequence_start), end_idx):
            current_price = data.iloc[i]['close']
            
            # 피처 및 시퀀스 준비
            if i >= len(features_df) or i - sequence_start < 0 or i - sequence_start >= len(sequences):
                continue
            
            test_features = features_df.iloc[i].values
            test_sequence = sequences[i - sequence_start].reshape(1, sequence_length, 5)
            current_state = {
                'current_price': float(current_price),
                'current_position': self.position,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'market_volatility': float(data.iloc[i].get('volatility', 0.01)) if 'volatility' in data.columns else 0.01
            }
            
            # 거래 신호 생성
            try:
                signal = ensemble.generate_signal(test_features, test_sequence, current_state)
                
                # 신호 기록 (모델별 기여도 분석용)
                price_pred = signal.get('reasoning', {}).get('price_prediction', {})
                self.signal_history.append({
                    'timestamp': data.index[i],
                    'price': current_price,
                    'action': signal['action'],
                    'xgb_buy_prob': signal['reasoning']['direction_signal'].get('buy_prob', 0),
                    'xgb_sell_prob': signal['reasoning']['direction_signal'].get('sell_prob', 0),
                    'xgb_hold_prob': signal['reasoning']['direction_signal'].get('hold_prob', 0),
                    'lstm_target': price_pred.get('target_price', 0.0),  # 분석용 (RL의 학습 자료)
                    'lstm_stop_loss': price_pred.get('stop_loss', 0.0),  # 분석용 (RL의 학습 자료)
                    'rl_position_size': signal['reasoning']['rl_decision'].get('position_size', 0.0),
                    'confidence': signal['confidence']
                })
                
                # 디버깅: 비중 변화가 있을 때만 출력
                # 신호 해석: 비중 변화가 BUY/SELL을 의미 (비중 증가 = BUY, 비중 감소 = SELL)
                old_position = self.position
                new_position = signal['size']
                position_change = abs(new_position - old_position)
                
                # 의미있는 변화가 있을 때만 출력 (10% 단위 이산화 고려)
                min_change_threshold = 0.05  # 5% 이상 변화 시에만 출력
                if position_change >= min_change_threshold:
                    signal_type = "BUY" if new_position > old_position else "SELL"
                    print(f"[샘플] 인덱스 {i}, 가격: {current_price:.2f}, 신호: {signal_type}, "
                          f"비중: {old_position:.2f} → {new_position:.2f}")
                # 처음 몇 개는 항상 출력 (디버깅용)
                elif i < start_idx + 5:
                    signal_type = "HOLD" if position_change < 1e-6 else ("BUY" if new_position > old_position else "SELL")
                    print(f"[샘플] 인덱스 {i}, 가격: {current_price:.2f}, 신호: {signal_type}, "
                          f"비중: {signal['size']:.2f}")
            except Exception as e:
                print(f"[샘플] 신호 생성 오류 (인덱스 {i}): {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # RL의 포지션 관리: RL이 결정한 action_type과 position_size에 따라 포지션 조절
            # 익절/손절 없이 비중으로만 리스크 관리
            self._adjust_position_by_rl(signal, current_price, i, data.index[i])
        
        # 최종 포지션 청산
        if self.position != 0.0:
            final_price = data.iloc[end_idx - 1]['close']
            self._close_position(final_price, end_idx - 1, data.index[end_idx - 1])
        
        # 성과 분석
        results = self._analyze_results()

        # 포트폴리오 표지용 차트 저장
        try:
            self._save_portfolio_cover_chart(
                price_series=data['close'].iloc[max(start_idx, sequence_start):end_idx],
                results=results
            )
        except Exception as _e:
            # 차트 저장 실패는 샘플 동작에 영향 주지 않음
            print(f"[샘플] 포트폴리오 표지 차트 저장 중 오류: {_e}")
        
        print("\n[샘플] 백테스트 완료")
        return results
    
    def _adjust_position_by_rl(self, signal: Dict, current_price: float, idx: int, timestamp):
        """RL의 포지션 관리: RL이 결정한 position_size에 따라 포지션 조절"""
        action = signal['action']
        size = signal['size']  # RL이 결정한 목표 포지션 크기
        rl_dec = signal.get('reasoning', {}).get('rl_decision', {})
        
        # 기존 포지션
        old_position = self.position
        
        # RL이 결정한 대로 포지션 조절 (position_size를 목표 포지션으로 직접 설정)
        # 주식은 숏 포지션이 없으므로 position_size만으로 충분
        # position_size: 0.0 (청산) ~ 1.0 (최대 롱)
        new_position = min(size, 1.0)  # 목표 포지션 크기
        
        # 포지션 변경 처리 (부동소수점 오차 고려)
        EPSILON = 1e-6  # 부동소수점 비교를 위한 작은 값
        
        # 실제 포지션 변화량 계산 (부동소수점 오차 고려)
        position_change = abs(new_position - old_position)
        
        # 의미있는 변화가 없으면 무시 (부동소수점 오차로 인한 미세한 변화 방지)
        if position_change < EPSILON:
            return
        
        if abs(new_position) < EPSILON and old_position > EPSILON:
            # 전체 청산
            self._close_position(current_price, idx, timestamp, reason='RL 포지션 조절 (전체 청산)')
        elif (old_position - new_position) > EPSILON and old_position > EPSILON:
            # 일부 청산 (포지션 감소) - 거래로 기록
            # 실제 청산 비중 계산
            closed_size = old_position - new_position
            # 손익 계산: (가격변화율) * (청산비중) * (현재자본)
            # closed_size는 비중 (0.01 = 1%), balance는 현재 자본
            if self.entry_price > 0:
                pnl = (current_price - self.entry_price) / self.entry_price * closed_size * self.balance
            else:
                pnl = 0.0
            
            # 거래 비용 차감 (RL 학습 환경과 동일한 방식)
            transaction_cost_rate = self.config.get('backtest', {}).get('transaction_cost_rate', 0.001)
            transaction_cost = closed_size * self.balance * transaction_cost_rate
            pnl -= transaction_cost
            
            self.balance += pnl
            
            # 일부 청산 거래 기록
            # 비중 감소 = SELL 신호이므로 entry_action을 'SELL'로 설정
            self._update_entry_meta(signal, 'SELL', current_price)
            trade_record = {
                'timestamp': timestamp,
                'action': 'PARTIAL_CLOSE',
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'position_size': closed_size,
                'pnl': pnl,
                'transaction_cost': transaction_cost,  # 거래 비용 기록
                'reason': f'일부 청산 ({old_position:.2f} → {new_position:.2f})',
                # 진입 시점 메타데이터 포함 (비중 감소 = SELL)
                'entry_action': 'SELL',
                'xgb_buy_prob_entry': self.entry_meta.get('xgb_buy_prob_entry', 0.0),
                'xgb_sell_prob_entry': self.entry_meta.get('xgb_sell_prob_entry', 0.0),
                'xgb_hold_prob_entry': self.entry_meta.get('xgb_hold_prob_entry', 0.0),
                'xgb_action_entry': self.entry_meta.get('xgb_action_entry', 'HOLD'),
                'lstm_target_entry': self.entry_meta.get('lstm_target_entry', current_price * 1.02),
                'lstm_stop_entry': self.entry_meta.get('lstm_stop_entry', current_price * 0.98),
                'rl_position_size': self.entry_meta.get('rl_position_size', 0.0)
            }
            self.trade_history.append(trade_record)
            
            self.position = new_position
            print(f"[샘플] RL 일부 청산: {timestamp}, 가격: {current_price:.2f}, 비중: {old_position:.2f} → {new_position:.2f}, 청산비중: {closed_size:.4f}, 손익: {pnl:.2f}")
        elif (new_position - old_position) > EPSILON:
            # 추가 매수 또는 새로 진입 - 거래로 기록
            # 비중 증가 = BUY 신호이므로 entry_action을 'BUY'로 설정
            self._update_entry_meta(signal, 'BUY', current_price)
            
            if old_position < EPSILON:
                # 새로 진입
                self.entry_price = current_price
                self.entry_timestamp = timestamp
                
                # 거래 비용 차감 (RL 학습 환경과 동일한 방식)
                transaction_cost_rate = self.config.get('backtest', {}).get('transaction_cost_rate', 0.001)
                transaction_cost = new_position * self.balance * transaction_cost_rate
                self.balance -= transaction_cost
            else:
                # 추가 매수 (평균 진입가 조정) - 거래로 기록
                additional_size = new_position - old_position
                
                # 거래 비용 차감 (RL 학습 환경과 동일한 방식)
                transaction_cost_rate = self.config.get('backtest', {}).get('transaction_cost_rate', 0.001)
                transaction_cost = additional_size * self.balance * transaction_cost_rate
                self.balance -= transaction_cost
                
                trade_record = {
                    'timestamp': timestamp,
                    'action': 'ADD_BUY',
                    'entry_price': current_price,
                    'exit_price': current_price,
                    'position_size': additional_size,
                    'pnl': 0.0,  # 추가 매수는 즉시 손익 없음
                    'transaction_cost': transaction_cost,  # 거래 비용 기록
                    'reason': f'추가 매수 ({old_position:.2f} → {new_position:.2f})',
                    # 진입 시점 메타데이터 포함 (비중 증가 = BUY)
                    'entry_action': 'BUY',
                    'xgb_buy_prob_entry': self.entry_meta.get('xgb_buy_prob_entry', 0.0),
                    'xgb_sell_prob_entry': self.entry_meta.get('xgb_sell_prob_entry', 0.0),
                    'xgb_hold_prob_entry': self.entry_meta.get('xgb_hold_prob_entry', 0.0),
                    'xgb_action_entry': self.entry_meta.get('xgb_action_entry', 'HOLD'),
                    'lstm_target_entry': self.entry_meta.get('lstm_target_entry', current_price * 1.02),
                    'lstm_stop_entry': self.entry_meta.get('lstm_stop_entry', current_price * 0.98),
                    'rl_position_size': self.entry_meta.get('rl_position_size', 0.0)
                }
                self.trade_history.append(trade_record)
                
                # 평균 진입가 조정
                total_value = self.entry_price * old_position + current_price * additional_size
                self.entry_price = total_value / new_position if new_position > EPSILON else current_price
            
            self.position = new_position
            print(f"[샘플] RL 매수: {timestamp}, 가격: {current_price:.2f}, 비중: {old_position:.2f} → {new_position:.2f}")
        # abs(new_position - old_position) <= EPSILON이면 변경 없음 (로그 출력 안 함)
    
    def _update_entry_meta(self, signal: Dict, action: str, current_price: float):
        """진입 메타데이터 업데이트"""
        xgb_sig = signal.get('reasoning', {}).get('direction_signal', {})
        rl_dec = signal.get('reasoning', {}).get('rl_decision', {})
        price_pred = signal.get('reasoning', {}).get('price_prediction', {})
        xgb_buy = float(xgb_sig.get('buy_prob', 0.0))
        xgb_sell = float(xgb_sig.get('sell_prob', 0.0))
        xgb_hold = float(xgb_sig.get('hold_prob', 0.0))
        xgb_action = 'BUY' if xgb_buy > max(xgb_sell, xgb_hold) else ('SELL' if xgb_sell > max(xgb_buy, xgb_hold) else 'HOLD')
        # LSTM 목표가/손절가는 분석용으로만 저장 (RL의 학습 자료)
        lstm_target = price_pred.get('target_price', current_price * 1.02)
        lstm_stop = price_pred.get('stop_loss', current_price * 0.98)
        self.entry_meta = {
            'entry_action': action,
            'xgb_buy_prob_entry': xgb_buy,
            'xgb_sell_prob_entry': xgb_sell,
            'xgb_hold_prob_entry': xgb_hold,
            'xgb_action_entry': xgb_action,
            'lstm_target_entry': float(lstm_target),  # 분석용 (RL의 학습 자료)
            'lstm_stop_entry': float(lstm_stop),  # 분석용 (RL의 학습 자료)
            'rl_position_size': float(rl_dec.get('position_size', 0.0)),
        }
    
    def _close_position(self, exit_price: float, idx: int, timestamp, reason: str = '청산'):
        """포지션 청산"""
        if self.position == 0.0:
            return
        
        # 손익 계산
        if self.entry_price > 0:
            if self.position > 0:  # 롱 포지션
                pnl = (exit_price - self.entry_price) / self.entry_price * abs(self.position) * self.balance
            else:  # 숏 포지션
                pnl = (self.entry_price - exit_price) / self.entry_price * abs(self.position) * self.balance
        else:
            pnl = 0.0
        
        # 거래 비용 (설정 파일에서 읽기)
        transaction_cost_rate = self.config.get('backtest', {}).get('transaction_cost_rate', 0.001)
        transaction_cost = abs(self.position) * self.balance * transaction_cost_rate
        pnl -= transaction_cost
        
        # 잔액 업데이트
        self.balance += pnl
        
        # 거래 기록
        trade_record = {
            'timestamp': timestamp,
            'entry_timestamp': self.entry_timestamp,
            'action': 'LONG_CLOSE' if self.position > 0 else 'SHORT_CLOSE',
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'size': abs(self.position),
            'pnl': pnl,
            'reason': reason,
            'balance_after': self.balance,
            # 진입 시점 메타 담기 (없을 경우 기본값)
            'entry_action': self.entry_meta.get('entry_action', 'HOLD'),
            'xgb_buy_prob_entry': self.entry_meta.get('xgb_buy_prob_entry', 0.0),
            'xgb_sell_prob_entry': self.entry_meta.get('xgb_sell_prob_entry', 0.0),
            'xgb_hold_prob_entry': self.entry_meta.get('xgb_hold_prob_entry', 0.0),
            'xgb_action_entry': self.entry_meta.get('xgb_action_entry', 'HOLD'),
            'lstm_target_entry': self.entry_meta.get('lstm_target_entry', 0.0),  # 분석용 (RL의 학습 자료)
            'lstm_stop_entry': self.entry_meta.get('lstm_stop_entry', 0.0),  # 분석용 (RL의 학습 자료)
            'rl_position_size': self.entry_meta.get('rl_position_size', 0.0),
        }
        self.trade_history.append(trade_record)
        
        print(f"[샘플] 포지션 청산: {reason}, 진입가: {self.entry_price:.2f}, 청산가: {exit_price:.2f}, 손익: {pnl:.2f}")
        
        # 포지션 초기화
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_timestamp = None
        # 익절/손절 제거: RL의 포지션 관리(비중)로만 리스크 관리
        self.entry_meta = {}
    
    def _analyze_results(self) -> Dict:
        """백테스트 결과 분석"""
        if not self.trade_history:
            # 거래가 없어도 기본 성과 지표 반환
            final_balance = self.balance
            total_return = (final_balance - self.initial_balance) / self.initial_balance
            return {
                'initial_balance': self.initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'trade_history': pd.DataFrame(),
                'message': '거래가 발생하지 않았습니다.'
            }
        
        df = pd.DataFrame(self.trade_history)
        
        # 기본 통계
        # 승률 계산: 부분 청산(PARTIAL_CLOSE) + 전체 청산(LONG_CLOSE) 모두 포함
        # 청산 시 수익이면 승리, 손실이면 패배
        close_trades = df[df['action'].isin(['PARTIAL_CLOSE', 'LONG_CLOSE'])]
        # pnl이 0이 아닌 거래만 포함 (실제 손익이 발생한 거래)
        close_trades_with_pnl = close_trades[close_trades['pnl'] != 0.0]
        total_trades = len(close_trades_with_pnl)
        profitable_trades = len(close_trades_with_pnl[close_trades_with_pnl['pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 전체 손익 통계 (PARTIAL_CLOSE + LONG_CLOSE 포함)
        trades_with_pnl = df[df['pnl'] != 0.0]  # 손익이 있는 모든 거래
        
        # 전체 거래 수 (분석용)
        total_all_trades = len(df)
        
        # 손익이 있는 거래만으로 통계 계산
        total_pnl = trades_with_pnl['pnl'].sum() if len(trades_with_pnl) > 0 else 0.0
        final_balance = self.balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        avg_pnl = trades_with_pnl['pnl'].mean() if len(trades_with_pnl) > 0 else 0.0
        max_profit = trades_with_pnl['pnl'].max() if len(trades_with_pnl) > 0 else 0.0
        max_loss = trades_with_pnl['pnl'].min() if len(trades_with_pnl) > 0 else 0.0
        
        # Sharpe Ratio (손익이 있는 거래만)
        returns = trades_with_pnl['pnl'].values if len(trades_with_pnl) > 0 else np.array([])
        sharpe_ratio = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 연율화
        
        # 최대 낙폭
        cumulative = pd.Series(returns).cumsum()  # pandas Series로 변환
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        max_drawdown_pct = max_drawdown / self.initial_balance if self.initial_balance > 0 else 0.0
        
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': total_trades,  # 청산 거래 수 (PARTIAL_CLOSE + LONG_CLOSE, pnl != 0)
            'total_all_trades': len(trades_with_pnl),  # 전체 거래 수 (모든 거래, pnl != 0)
            'profitable_trades': profitable_trades,  # 청산 승리 거래 수 (PARTIAL_CLOSE + LONG_CLOSE, pnl > 0)
            'win_rate': win_rate,  # 청산 승률 (PARTIAL_CLOSE + LONG_CLOSE)
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'trade_history': df
        }
        
        # 모델별 기여도 분석 제거: 최종 결정은 RL Agent가 내리므로 개별 모델 분석은 의미 없음
        # XGBoost와 LSTM은 RL Agent의 입력 정보로만 사용되며, 최종 포지션 크기 결정은 RL Agent가 수행
        
        return results

    def _save_portfolio_cover_chart(self, price_series: pd.Series, results: Dict):
        """최종 트레이딩 결과를 차트로 시각화하여 저장 (포트폴리오 표지용)"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams
        from matplotlib import dates as mdates
        from matplotlib.patches import FancyBboxPatch
        import platform
        import numpy as _np
        import os as _os

        # 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: DejaVu/NanumGothic)
        def _set_korean_font():
            system = platform.system()
            candidates = []
            if system == 'Windows':
                candidates = ['Malgun Gothic', '맑은 고딕', 'NanumGothic', 'DejaVu Sans']
            elif system == 'Darwin':
                candidates = ['AppleGothic', 'NanumGothic', 'DejaVu Sans']
            else:
                candidates = ['NanumGothic', 'DejaVu Sans']
            for name in candidates:
                try:
                    rcParams['font.family'] = name
                    # 테스트 렌더링
                    return
                except Exception:
                    continue
            # 마지막 대비
            rcParams['font.family'] = 'DejaVu Sans'
        _set_korean_font()
        rcParams['axes.unicode_minus'] = False

        # 정사각 480x480px: 4.8인치 * 100dpi
        fig, ax = plt.subplots(figsize=(4.8, 4.8))

        # 가격 라인
        ax.plot(price_series.index, price_series.values, color='black', linewidth=1.2, label='종가(Price)')

        # 청산 마커 및 진입-청산 연결선 (trade_history 기반) - 표지에는 청산 위주 표시
        if len(self.trade_history) > 0:
            th = pd.DataFrame(self.trade_history)
            th = th[(th['timestamp'] >= price_series.index[0]) & (th['timestamp'] <= price_series.index[-1])]
            # 익절: 초록 점, 손절: 빨강 X
            tp = th[th['reason'] == '익절']
            sl = th[th['reason'] == '손절']
            if len(tp) > 0:
                ax.scatter(tp['timestamp'], tp['exit_price'], marker='o', color='tab:green', s=20, label='익절(청산)')
            if len(sl) > 0:
                ax.scatter(sl['timestamp'], sl['exit_price'], marker='x', color='tab:red', s=28, label='손절(청산)')
            # 진입-청산 라인 (수익: 녹색, 손실: 붉은색)
            for _, tr in th.iterrows():
                et = tr.get('entry_timestamp', None)
                if et is None:
                    continue
                try:
                    x_vals = [et, tr['timestamp']]
                    y_vals = [tr['entry_price'], tr['exit_price']]
                    color = 'tab:green' if tr['pnl'] > 0 else 'tab:red'
                    ax.plot(x_vals, y_vals, color=color, alpha=0.6, linewidth=1.2)
                except Exception:
                    continue

        ax.grid(True, alpha=0.25)

        # X축 날짜 눈금 밀집 완화
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        formatter = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)

        # 성과 지표
        total_return = results.get('total_return', 0.0)
        win_rate = results.get('win_rate', 0.0)
        sharpe = results.get('sharpe_ratio', 0.0)
        total_trades = results.get('total_trades', 0)
        final_balance = results.get('final_balance', 0.0)
        avg_pnl = results.get('avg_pnl', 0.0)
        max_profit = results.get('max_profit', 0.0)
        max_loss = results.get('max_loss', 0.0)
        max_dd = results.get('max_drawdown', 0.0)
        max_dd_pct = results.get('max_drawdown_pct', 0.0)

        # 모델 분석
        # 모델별 기여도 분석 제거: 최종 결정은 RL Agent가 내리므로 개별 모델 분석은 의미 없음

        # 하단 정보 박스 (요구 항목만 표기: 총 수익률, 승률, Sharpe, 최대 낙폭)
        footer_ax = fig.add_axes([0.03, 0.03, 0.94, 0.14])
        footer_ax.axis('off')
        bg = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=6",
                            facecolor='#F9FAFB', edgecolor='#E5E7EB', linewidth=1.0, transform=footer_ax.transAxes)
        footer_ax.add_patch(bg)
        # 제목
        footer_ax.text(0.02, 0.68, "성과 요약", fontsize=9, weight='bold', color='#111827', va='center', ha='left')
        # 핵심 좌측: 총 수익률, 승률, Sharpe
        footer_ax.text(0.02, 0.36, f"총 수익률 {total_return:.2%}   승률 {win_rate:.2%}   Sharpe {sharpe:.2f}",
                       fontsize=8.5, color='#111827', va='center', ha='left')
        # 핵심 우측: 최대 낙폭
        footer_ax.text(0.98, 0.36, f"최대 낙폭 {max_dd:,.0f} ({max_dd_pct:.2%})",
                       fontsize=8.5, color='#111827', va='center', ha='right')

        # 간단한 범례 (청산 마커만)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc='upper left', fontsize=8, frameon=True)
            leg.get_frame().set_alpha(0.85)

        ax.set_title('샘플 백테스트 결과', fontsize=11, pad=8)
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        plt.tight_layout(rect=[0, 0.18, 1, 1])  # 하단 정보 박스 공간 확보

        # 저장 경로
        example_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        save_dir = _os.path.join(example_dir, 'results')
        _os.makedirs(save_dir, exist_ok=True)
        # 더 직관적인 파일명
        save_path = _os.path.join(save_dir, 'ai_trading_cover_480.png')
        plt.savefig(save_path, dpi=100)  # 4.8in * 100dpi = 480px
        plt.close(fig)
        print(f"[샘플] 포트폴리오 표지 차트 저장: {save_path}")

