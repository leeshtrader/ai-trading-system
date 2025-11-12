"""
기술적 지표 생성 모듈
기술적 지표를 계산하고 피처를 생성합니다.

주의: 본 모듈은 샘플/데모 목적입니다.

Technical Indicator Generation Module
Calculates technical indicators and generates features for ML models.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class FeatureEngineering:
    """
    피처 엔지니어링 모듈
    기술적 지표를 계산하고 AI 모델에 필요한 피처를 생성합니다.
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.fe_config = self.config.get('feature_engineering', {})
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 계산
        
        추세 추종 지표:
        - MA (Moving Average)
        - EMA (Exponential Moving Average)
        - MACD
        
        평균회귀 지표:
        - RSI
        - Bollinger Bands
        - Z-Score
        """
        df = data.copy()
        
        # 설정 파일에서 윈도우 크기 읽기
        ma_5 = self.fe_config.get('ma_5', 5)
        ma_20 = self.fe_config.get('ma_20', 20)
        ma_60 = self.fe_config.get('ma_60', 60)
        ema_12 = self.fe_config.get('ema_12', 12)
        ema_26 = self.fe_config.get('ema_26', 26)
        macd_signal_window = self.fe_config.get('macd_signal_window', 9)
        rsi_window = self.fe_config.get('rsi_window', 14)
        bb_window = self.fe_config.get('bb_window', 20)
        bb_std = self.fe_config.get('bb_std', 2)
        z_score_window = self.fe_config.get('z_score_window', 20)
        volume_ma_window = self.fe_config.get('volume_ma_window', 20)
        volatility_window = self.fe_config.get('volatility_window', 20)
        
        # 추세 추종 지표
        df['ma_5'] = df['close'].rolling(window=ma_5).mean()
        df['ma_20'] = df['close'].rolling(window=ma_20).mean()
        df['ma_60'] = df['close'].rolling(window=ma_60).mean()
        df['ema_12'] = df['close'].ewm(span=ema_12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=ema_26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=macd_signal_window, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI (평균회귀)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (평균회귀)
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_val = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Z-Score (평균회귀)
        df['z_score'] = (df['close'] - df['close'].rolling(window=z_score_window).mean()) / df['close'].rolling(window=z_score_window).std()
        
        # 거래량 지표
        df['volume_ma'] = df['volume'].rolling(window=volume_ma_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 가격 변화율
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=volatility_window).std()
        
        return df
    
    def prepare_features_for_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        XGBoost용 구조화된 피처 생성
        """
        features = pd.DataFrame({
            # 추세 지표
            'ma_5_ratio': df['close'] / df['ma_5'] - 1,
            'ma_20_ratio': df['close'] / df['ma_20'] - 1,
            'ma_60_ratio': df['close'] / df['ma_60'] - 1,
            'macd': df['macd'],
            'macd_signal': df['macd_signal'],
            'macd_hist': df['macd_hist'],
            
            # 평균회귀 지표
            'rsi': df['rsi'] / 100,  # 0-1로 정규화
            'bb_position': df['bb_position'],  # 0-1
            'bb_width': df['bb_width'],
            'z_score': df['z_score'],
            
            # 거래량 및 변동성
            'volume_ratio': df['volume_ratio'],
            'volatility': df['volatility'],
            'price_change': df['price_change'],
        })
        
        return features
    
    def prepare_sequences_for_lstm(self, df: pd.DataFrame, sequence_length: int = 20) -> np.ndarray:
        """
        LSTM용 시계열 데이터 생성
        """
        # 정규화된 가격 및 지표 시퀀스
        sequences = []
        
        lstm_sequence_features = self.fe_config.get('lstm_sequence_features', ['close', 'volume', 'rsi', 'macd', 'bb_position'])
        for i in range(sequence_length, len(df)):
            seq = df.iloc[i-sequence_length:i][lstm_sequence_features].values
            
            # 정규화 (각 컬럼별로)
            seq_normalized = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
            sequences.append(seq_normalized)
        
        return np.array(sequences)
    
    def get_current_state(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        현재 상태 정보 생성
        강화학습에 필요한 현재 상태
        """
        current_price = df.iloc[idx]['close']
        
        default_initial_balance = self.fe_config.get('default_initial_balance', 100000.0)
        default_initial_position = self.fe_config.get('default_initial_position', 0.0)
        default_volatility = self.fe_config.get('default_volatility', 0.01)
        
        return {
            'current_price': float(current_price),
            'current_position': default_initial_position,
            'balance': default_initial_balance,
            'unrealized_pnl': 0.0,
            'market_volatility': float(df.iloc[idx]['volatility']) if not pd.isna(df.iloc[idx]['volatility']) else default_volatility
        }

