"""
샘플: AI Learning Module 학습 스크립트
XGBoost, LSTM, 강화학습 모델을 학습합니다.

주의: 본 스크립트는 샘플/데모 목적입니다.

Sample: AI Learning Module Training Script
Trains XGBoost, LSTM, and Reinforcement Learning models.

WARNING: This script is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import sys
import os
import pandas as pd
import numpy as np

# 경로 설정
# train_sample.py는 example/ai_learning/에 있으므로
# example_dir는 두 단계 위로 올라가야 함
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import get_data_config, get_training_config, get_labeling_config, get_backtest_config
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel
from ai_learning.models.lstm_price import LSTMPriceModel
from ai_learning.models.rl_agent import TradingRLAgent
from ai_learning.ensemble.role_based_ensemble import RoleBasedEnsemble

# backtest는 같은 디렉토리에 있으므로 상대 import
try:
    from backtest import Backtester
except ImportError:
    # 직접 경로로 import
    import importlib.util
    backtest_path = os.path.join(os.path.dirname(__file__), 'backtest.py')
    spec = importlib.util.spec_from_file_location("backtest", backtest_path)
    backtest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backtest_module)
    Backtester = backtest_module.Backtester

def load_data(config):
    """데이터 로드"""
    # train_sample.py는 example/ai_learning/에 있으므로
    # example_dir는 두 단계 위로 올라가야 함
    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 설정 파일에서 데이터 경로 읽기
    data_config = get_data_config(config)
    symbol = data_config.get('symbol', 'NVDA')
    output_dir = data_config.get('output_dir', 'data')
    output_file = data_config.get('output_file', f'{symbol.lower()}_data.csv')
    
    data_path = os.path.join(example_dir, output_dir, output_file)
    
    if not os.path.exists(data_path):
        print(f"[샘플] 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("[샘플] data/download_data.py를 먼저 실행하세요.")
        return None
    
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[샘플] 데이터 로드 완료: {len(data)}개 데이터 포인트 (경로: {data_path})")
    return data

def create_labels(features_df: pd.DataFrame, price_data: pd.Series, config) -> np.ndarray:
    """
    샘플: 간단한 레이블 생성
    """
    labeling_config = get_labeling_config(config)
    buy_percentile = labeling_config.get('buy_percentile', 70)
    sell_percentile = labeling_config.get('sell_percentile', 30)
    
    labels = []
    price_changes = []
    
    # 먼저 모든 가격 변화율 계산
    for i in range(len(features_df) - 1):
        current_price = price_data.iloc[i]
        next_price = price_data.iloc[i + 1]
        price_change = (next_price - current_price) / current_price
        price_changes.append(price_change)
    
    # 설정 파일에서 읽은 퍼센타일로 BUY/SELL 분류
    if len(price_changes) > 0:
        price_changes_array = np.array(price_changes)
        buy_threshold = np.percentile(price_changes_array, buy_percentile)
        sell_threshold = np.percentile(price_changes_array, sell_percentile)
        
        for price_change in price_changes:
            if price_change >= buy_threshold:
                labels.append(1)  # BUY
            elif price_change <= sell_threshold:
                labels.append(2)  # SELL
            else:
                labels.append(0)  # HOLD
    else:
        labels = [0] * len(features_df)
    
    # 마지막 데이터는 레이블이 없으므로 HOLD
    labels.append(0)
    
    return np.array(labels)

def create_price_targets(price_data: pd.Series, config) -> np.ndarray:
    """
    샘플: 가격 목표 생성
    """
    labeling_config = get_labeling_config(config)
    target_ratio = labeling_config.get('default_target_ratio', 1.02)
    stop_loss_ratio = labeling_config.get('default_stop_loss_ratio', 0.98)
    time_horizon = labeling_config.get('default_time_horizon', 5)
    confidence = labeling_config.get('default_confidence', 0.7)
    
    targets = []
    
    for i in range(len(price_data) - 1):
        current_price = price_data.iloc[i]
        next_price = price_data.iloc[i + 1]
        
        # 설정 파일에서 읽은 비율로 목표가 설정
        target_price = next_price * target_ratio
        stop_loss = current_price * stop_loss_ratio
        volatility = abs((next_price - current_price) / current_price)
        
        targets.append([target_price, stop_loss, time_horizon, confidence, volatility])
    
    # 마지막 데이터
    last_price = price_data.iloc[-1]
    targets.append([last_price * target_ratio, last_price * stop_loss_ratio, time_horizon, confidence, 0.01])
    
    return np.array(targets)

def main(config=None):
    """샘플 학습 메인 함수"""
    # config가 없으면 기본 설정 로드
    if config is None:
        from config.config_loader import load_config
        config = load_config()
    
    print("=" * 50)
    print("[샘플] AI Learning Module 학습 시작")
    print("=" * 50)
    
    # 설정 읽기
    training_config = get_training_config(config)
    train_test_split = training_config.get('train_test_split', 0.8)
    validation_split = training_config.get('validation_split', 0.8)
    
    # 1. 데이터 로드
    data = load_data(config)
    if data is None:
        return None
    
    # 2. 피처 엔지니어링 (기술적 지표 생성)
    print("\n[샘플] 피처 엔지니어링: 기술적 지표 생성")
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    
    # 결측값 제거
    data_with_indicators = data_with_indicators.dropna()
    print(f"[샘플] 기술적 지표 생성 완료: {len(data_with_indicators)}개 데이터 포인트")
    
    # 3. 피처 준비
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    sequences = feature_eng.prepare_sequences_for_lstm(data_with_indicators, sequence_length=sequence_length)
    
    # 레이블 생성 (샘플용)
    labels = create_labels(features_df, data_with_indicators['close'], config)
    price_targets = create_price_targets(data_with_indicators['close'], config)
    
    # 학습/테스트 분할
    split_idx = int(len(features_df) * train_test_split)
    
    X_train = features_df.iloc[:split_idx].values
    X_test = features_df.iloc[split_idx:].values
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    
    seq_train = sequences[:split_idx-sequence_length]  # 시퀀스는 sequence_length개 앞서 시작
    seq_test = sequences[split_idx-sequence_length:]
    target_train = price_targets[:split_idx-sequence_length]
    target_test = price_targets[split_idx-sequence_length:]
    
    # 4. XGBoost 학습
    print("\n[샘플] XGBoost 학습 시작")
    xgb_config = config.get('models', {}).get('xgboost', {})
    xgb_model = XGBoostDirectionModel(xgb_config)
    
    # 검증 데이터 분할
    val_split_idx = int(len(X_train) * validation_split)
    X_train_split = X_train[:val_split_idx]
    X_val_split = X_train[val_split_idx:]
    y_train_split = y_train[:val_split_idx]
    y_val_split = y_train[val_split_idx:]
    
    xgb_model.train(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # XGBoost 학습 결과 시각화
    feature_names = list(features_df.columns)
    save_dir = os.path.join(example_dir, 'results', 'xgboost')
    xgb_model.visualize_results(X_test, y_test, feature_names=feature_names, save_dir=save_dir)
    
    # 5. LSTM 학습
    print("\n[샘플] LSTM 학습 시작")
    lstm_model = LSTMPriceModel(lstm_config)
    input_size = lstm_config.get('input_size', 5)
    lstm_model.build_model(input_size=input_size)
    
    # 검증 데이터 분할
    val_split_idx = int(len(seq_train) * validation_split)
    seq_train_split = seq_train[:val_split_idx]
    seq_val_split = seq_train[val_split_idx:]
    target_train_split = target_train[:val_split_idx]
    target_val_split = target_train[val_split_idx:]
    
    epochs = lstm_config.get('epochs', 30)
    lstm_model.train(seq_train_split, target_train_split, 
                     sequences_val=seq_val_split, 
                     targets_val=target_val_split,
                     epochs=epochs)
    
    # LSTM 학습 결과 시각화
    save_dir = os.path.join(example_dir, 'results', 'lstm')
    lstm_model.visualize_results(seq_test, target_test, save_dir=save_dir)
    
    # 6. 강화학습 학습 (선택사항 - stable-baselines3가 설치된 경우만)
    print("\n[샘플] 강화학습 학습 시작")
    try:
        price_array = data_with_indicators['close'].values
        rl_config = config.get('models', {}).get('rl', {})
        rl_agent = TradingRLAgent(rl_config)
        rl_agent.train(price_array)
        
        # RL 시각화 저장
        save_dir = os.path.join(example_dir, 'results', 'rl')
        rl_agent.visualize_results(price_array, save_dir=save_dir)
    except Exception as e:
        print(f"[샘플] 강화학습 학습 건너뜀: {e}")
        print("[샘플] 기본 RL Agent 사용")
        rl_config = config.get('models', {}).get('rl', {})
        rl_agent = TradingRLAgent(rl_config)
    
    # 7. 앙상블 생성
    print("\n[샘플] 역할 분담 앙상블 생성")
    try:
        ensemble = RoleBasedEnsemble(xgb_model, lstm_model, rl_agent, config)
        
        # 8. 샘플 예측 테스트
        print("\n[샘플] 샘플 예측 테스트")
        if len(seq_test) > 0:
            test_idx = 0
            test_features = X_test[test_idx]
            test_sequence = seq_test[test_idx].reshape(1, sequence_length, 5)
            test_state = feature_eng.get_current_state(data_with_indicators, split_idx + test_idx)
            
            signal = ensemble.generate_signal(test_features, test_sequence, test_state)
            
            print("\n[샘플] 생성된 거래 신호:")
            print(f"  Action: {signal['action']}")
            print(f"  Size: {signal['size']:.2f}")
            print(f"  Target Price: {signal['target_price']:.2f}")
            print(f"  Stop Loss: {signal['stop_loss']:.2f}")
            print(f"  Confidence: {signal['confidence']:.2f}")
        else:
            print("[샘플] 테스트 데이터가 부족하여 예측을 건너뜁니다.")
        
        # 9. 백테스트 실행
        print("\n[샘플] 백테스트 실행 (테스트 데이터)")
        backtest_config = get_backtest_config(config)
        initial_balance = backtest_config.get('initial_balance', 100000)
        backtester = Backtester(initial_balance=initial_balance, config=config)
        backtest_results = backtester.run_backtest(
            data=data_with_indicators,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            rl_agent=rl_agent,
            feature_eng=feature_eng,
            start_idx=split_idx,
            end_idx=len(data_with_indicators)
        )
        
        # 백테스트 결과를 반환값으로 포함
        return {
            'xgb_model': xgb_model,
            'lstm_model': lstm_model,
            'rl_agent': rl_agent,
            'ensemble': ensemble,
            'backtest_results': backtest_results
        }
        
    except Exception as e:
        print(f"[샘플] 앙상블 생성/예측 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'xgb_model': xgb_model,
            'lstm_model': lstm_model,
            'rl_agent': rl_agent,
            'ensemble': None,
            'backtest_results': None
        }
    
    print("\n" + "=" * 50)
    print("[샘플] AI Learning Module 학습 완료")
    print("=" * 50)

if __name__ == "__main__":
    main()

