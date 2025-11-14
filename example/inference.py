"""
샘플: 추론 스크립트
저장된 모델을 로드하여 새로운 데이터에 대한 예측을 수행합니다.

주의: 본 스크립트는 샘플/데모 목적입니다.

Sample: Inference Script
Loads saved models and performs predictions on new data.

WARNING: This script is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 경로 설정
example_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, example_dir)

from config.config_loader import load_config, get_data_config, get_training_config, get_model_config
from data.download_data import download_stock_data
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel
from ai_learning.models.lstm_price import LSTMPriceModel
from ai_learning.models.rl_agent import TradingRLAgent
from ai_learning.ensemble.role_based_ensemble import RoleBasedEnsemble


def load_trained_models(config):
    """
    저장된 모델들을 로드합니다.
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        로드된 모델들 (xgb_model, lstm_model, rl_agent)
    """
    training_config = get_training_config(config)
    models_dir = training_config.get('models_dir', 'models')
    models_path = os.path.join(example_dir, models_dir)
    
    print(f"[추론] 모델 로드 경로: {models_path}")
    
    # XGBoost 모델 로드
    xgb_path = os.path.join(models_path, 'xgboost_model.pkl')
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost 모델을 찾을 수 없습니다: {xgb_path}\n먼저 train_sample.py를 실행하여 모델을 학습하고 저장하세요.")
    
    xgb_config = get_model_config(config, 'xgboost')
    xgb_model = XGBoostDirectionModel(xgb_config)
    xgb_model.load(xgb_path)
    print("[추론] XGBoost 모델 로드 완료")
    
    # LSTM 모델 로드
    lstm_path = os.path.join(models_path, 'lstm_model.pth')
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM 모델을 찾을 수 없습니다: {lstm_path}\n먼저 train_sample.py를 실행하여 모델을 학습하고 저장하세요.")
    
    lstm_config = get_model_config(config, 'lstm')
    lstm_model = LSTMPriceModel(lstm_config)
    input_size = lstm_config.get('input_size', 5)
    lstm_model.load(lstm_path, input_size=input_size)
    print("[추론] LSTM 모델 로드 완료")
    
    # RL Agent 로드 (선택사항)
    rl_path = os.path.join(models_path, 'rl_agent.zip')
    rl_config = get_model_config(config, 'rl')
    rl_agent = TradingRLAgent(rl_config)
    
    if os.path.exists(rl_path):
        try:
            rl_agent.load(rl_path)
            print("[추론] RL Agent 로드 완료")
        except Exception as e:
            print(f"[추론] RL Agent 로드 실패 (기본 Agent 사용): {e}")
    else:
        print("[추론] RL Agent 파일이 없습니다. 기본 Agent를 사용합니다.")
    
    return xgb_model, lstm_model, rl_agent


def predict_single(idx: int, 
                   features_df: pd.DataFrame,
                   sequences: np.ndarray,
                   ensemble: RoleBasedEnsemble,
                   feature_eng: FeatureEngineering,
                   data: pd.DataFrame,
                   sequence_length: int) -> dict:
    """
    단일 데이터 포인트에 대한 예측을 수행합니다.
    
    Args:
        idx: 예측할 인덱스
        features_df: XGBoost용 피처 데이터프레임 (이미 준비됨)
        sequences: LSTM용 시퀀스 배열 (이미 준비됨)
        ensemble: 역할 분담 앙상블 객체 (재사용)
        feature_eng: 피처 엔지니어링 모듈
        data: 기술적 지표가 포함된 데이터프레임
        sequence_length: LSTM 시퀀스 길이
    
    Returns:
        예측 결과 딕셔너리
    """
    # 인덱스 검증
    if idx >= len(features_df):
        raise ValueError(f"인덱스 {idx}가 데이터 범위를 벗어났습니다. (최대: {len(features_df)-1})")
    
    if idx < sequence_length:
        raise ValueError(f"인덱스 {idx}는 시퀀스 길이({sequence_length})보다 작아야 합니다.")
    
    # XGBoost 피처
    xgb_features = features_df.iloc[idx].values
    
    # LSTM 시퀀스
    seq_idx = idx - sequence_length
    if seq_idx < 0 or seq_idx >= len(sequences):
        raise ValueError(f"시퀀스 인덱스 {seq_idx}가 범위를 벗어났습니다.")
    lstm_sequence = sequences[seq_idx].reshape(1, sequence_length, 5)
    
    # 현재 상태
    current_state = feature_eng.get_current_state(data, idx)
    
    # 앙상블을 사용하여 예측 (재사용)
    signal = ensemble.generate_signal(xgb_features, lstm_sequence, current_state)
    
    return signal


def predict_batch(data: pd.DataFrame,
                  ensemble: RoleBasedEnsemble,
                  feature_eng: FeatureEngineering,
                  config,
                  start_idx: int = None,
                  end_idx: int = None) -> list:
    """
    여러 데이터 포인트에 대한 배치 예측을 수행합니다.
    
    Args:
        data: 기술적 지표가 포함된 데이터프레임
        ensemble: 역할 분담 앙상블 객체 (재사용)
        feature_eng: 피처 엔지니어링 모듈
        config: 설정 딕셔너리
        start_idx: 시작 인덱스 (None이면 sequence_length부터)
        end_idx: 종료 인덱스 (None이면 끝까지)
    
    Returns:
        예측 결과 리스트
    """
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    
    # 피처 준비 (한 번만 수행)
    features_df = feature_eng.prepare_features_for_xgboost(data)
    sequences = feature_eng.prepare_sequences_for_lstm(data, sequence_length=sequence_length)
    
    if start_idx is None:
        start_idx = sequence_length
    if end_idx is None:
        end_idx = len(data)
    
    results = []
    for idx in range(start_idx, end_idx):
        try:
            signal = predict_single(
                idx, features_df, sequences, ensemble, 
                feature_eng, data, sequence_length
            )
            signal['timestamp'] = data.index[idx]
            signal['current_price'] = data.iloc[idx]['close']
            results.append(signal)
        except Exception as e:
            print(f"[추론] 인덱스 {idx} 예측 실패: {e}")
            continue
    
    return results


def main():
    """추론 메인 함수"""
    print("=" * 60)
    print("[추론] AI 트레이딩 시스템 추론 스크립트")
    print("=" * 60)
    
    # 설정 로드
    config = load_config()
    
    # 모델 로드
    print("\n[추론] 저장된 모델 로드 중...")
    try:
        xgb_model, lstm_model, rl_agent = load_trained_models(config)
    except FileNotFoundError as e:
        print(f"\n[추론] 오류: {e}")
        print("[추론] 먼저 train_sample.py를 실행하여 모델을 학습하고 저장하세요.")
        return
    
    # 데이터 로드
    print("\n[추론] 데이터 로드 중...")
    data_config = get_data_config(config)
    symbol = data_config.get('symbol', 'NVDA')
    output_dir = data_config.get('output_dir', 'data')
    output_file = data_config.get('output_file', f'{symbol.lower()}_data.csv')
    data_path = os.path.join(example_dir, output_dir, output_file)
    
    start_date = data_config.get('start_date', '2019-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    
    # 오늘 날짜 확인
    today = datetime.now().date()
    
    # 데이터 파일이 있는지 확인하고, 최신 데이터인지 확인
    need_download = False
    if not os.path.exists(data_path):
        print(f"[추론] 데이터 파일을 찾을 수 없습니다: {data_path}")
        need_download = True
    else:
        # 기존 데이터 로드하여 마지막 날짜 확인
        existing_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        last_date = existing_data.index[-1].date()
        
        # 마지막 날짜가 오늘보다 오래되었거나, 설정된 end_date보다 오래되었으면 다운로드
        if last_date < today:
            print(f"[추론] 기존 데이터의 마지막 날짜: {last_date}")
            print(f"[추론] 오늘 날짜: {today}")
            print(f"[추론] 최신 데이터를 다운로드합니다...")
            need_download = True
        else:
            print(f"[추론] 기존 데이터 사용 (마지막 날짜: {last_date})")
            data = existing_data
    
    if need_download:
        data = download_stock_data(symbol, start_date, end_date)
        # 데이터 저장
        from data.download_data import save_data
        save_data(data, data_path)
    
    print(f"[추론] 데이터 로드 완료: {len(data)}개 데이터 포인트")
    print(f"[추론] 데이터 기간: {data.index[0].date()} ~ {data.index[-1].date()}")
    
    # 피처 엔지니어링
    print("\n[추론] 피처 엔지니어링 중...")
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    print(f"[추론] 피처 엔지니어링 완료: {len(data_with_indicators)}개 데이터 포인트")
    
    # 앙상블 생성 (한 번만 생성)
    print("\n[추론] 앙상블 초기화 중...")
    ensemble = RoleBasedEnsemble(xgb_model, lstm_model, rl_agent, config)
    print("[추론] 앙상블 초기화 완료")
    
    # 예측 수행
    print("\n[추론] 예측 수행 중...")
    
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    
    # 가장 마지막 날짜에 대한 예측 (메인)
    last_idx = len(data_with_indicators) - 1
    if last_idx < sequence_length:
        print(f"[추론] 경고: 데이터가 부족합니다 (최소 {sequence_length}개 필요, 현재 {len(data_with_indicators)}개)")
        return None
    
    # 데이터 기준일 (가장 최신 거래일)
    data_date = data_with_indicators.index[last_idx]
    data_date_str = data_date.date() if isinstance(data_date, pd.Timestamp) else data_date
    
    # 다음 거래일 계산 (pandas BDay 사용)
    try:
        from pandas.tseries.offsets import BDay
        next_trading_day = (pd.Timestamp(data_date) + BDay(1)).date()
    except:
        # BDay를 사용할 수 없으면 단순히 +1일
        if isinstance(data_date, pd.Timestamp):
            next_trading_day = (data_date + pd.Timedelta(days=1)).date()
        else:
            next_trading_day = (pd.Timestamp(data_date) + pd.Timedelta(days=1)).date()
    
    print(f"[추론] 데이터 기준일: {data_date_str}")
    print(f"[추론] 예측 대상일 (다음 거래일): {next_trading_day}")
    
    # 피처 준비
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    sequences = feature_eng.prepare_sequences_for_lstm(data_with_indicators, sequence_length=sequence_length)
    
    # 최신 날짜 예측
    latest_signal = predict_single(
        last_idx, features_df, sequences, ensemble,
        feature_eng, data_with_indicators, sequence_length
    )
    latest_signal['data_date'] = data_date  # 데이터 기준일
    latest_signal['prediction_date'] = next_trading_day  # 예측 대상일 (다음 거래일)
    latest_signal['timestamp'] = data_date  # 호환성을 위해 유지
    latest_signal['current_price'] = data_with_indicators.iloc[last_idx]['close']
    
    # 결과 출력 (다음 거래일 예측 메인)
    print("\n" + "=" * 60)
    print("[추론] 다음 거래일 예측 결과 (메인)")
    print("=" * 60)
    print(f"\n데이터 기준일: {data_date_str}")
    print(f"예측 대상일: {next_trading_day} (다음 거래일)")
    print(f"기준일 종가: {latest_signal['current_price']:.2f}")
    print(f"행동: {latest_signal['action']}")
    print(f"포지션 크기: {latest_signal['size']:.2f}")
    print(f"목표가: {latest_signal['target_price']:.2f}")
    print(f"손절가: {latest_signal['stop_loss']:.2f}")
    print(f"신뢰도: {latest_signal['confidence']:.3f}")
    
    # 최근 10개 예측 (참고용)
    print("\n" + "=" * 60)
    print("[추론] 최근 10개 예측 (참고용)")
    print("=" * 60)
    start_idx = max(sequence_length, len(data_with_indicators) - 10)
    end_idx = len(data_with_indicators)
    
    recent_results = predict_batch(
        data_with_indicators,
        ensemble,
        feature_eng,
        config,
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    for i, result in enumerate(recent_results):
        is_latest = (result['timestamp'] == latest_signal['timestamp'])
        marker = " <-- 최신 (다음 거래일 예측)" if is_latest else ""
        data_date = result['timestamp'].date() if isinstance(result['timestamp'], pd.Timestamp) else result['timestamp']
        
        # 다음 거래일 계산
        try:
            from pandas.tseries.offsets import BDay
            pred_date = (pd.Timestamp(result['timestamp']) + BDay(1)).date()
        except:
            if isinstance(result['timestamp'], pd.Timestamp):
                pred_date = (result['timestamp'] + pd.Timedelta(days=1)).date()
            else:
                pred_date = (pd.Timestamp(result['timestamp']) + pd.Timedelta(days=1)).date()
        
        print(f"\n[{i+1}] 데이터 기준일: {data_date} → 예측일: {pred_date}{marker}")
        print(f"  기준일 종가: {result['current_price']:.2f}")
        print(f"  행동: {result['action']}")
        print(f"  포지션 크기: {result['size']:.2f}")
        print(f"  목표가: {result['target_price']:.2f}")
        print(f"  손절가: {result['stop_loss']:.2f}")
        print(f"  신뢰도: {result['confidence']:.3f}")
    
    # 백테스트 (최근 성과 확인용)
    backtest_results = None
    print("\n" + "=" * 60)
    print("[추론] 최근 백테스트 (성과 확인)")
    print("=" * 60)
    try:
        from ai_learning.backtest import Backtester
        from config.config_loader import get_backtest_config
        
        backtest_config = get_backtest_config(config)
        initial_balance = backtest_config.get('initial_balance', 100000)
        backtester = Backtester(initial_balance=initial_balance, config=config)
        
        # 최근 90일 또는 전체 데이터 중 작은 값으로 백테스트
        # 참고: 30일은 너무 짧아 통계적으로 의미가 없을 수 있음 (거래 횟수 부족)
        # 90일(약 3개월)은 최소한의 통계적 신뢰도를 제공
        backtest_days = min(90, len(data_with_indicators) - sequence_length)
        backtest_start_idx = max(sequence_length, len(data_with_indicators) - backtest_days)
        
        print(f"[추론] 최근 {backtest_days}일 백테스트 실행 중...")
        backtest_results = backtester.run_backtest(
            data=data_with_indicators,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            rl_agent=rl_agent,
            feature_eng=feature_eng,
            start_idx=backtest_start_idx,
            end_idx=len(data_with_indicators)
        )
        
        if backtest_results:
            print(f"\n[추론] 백테스트 결과:")
            print(f"  초기 자본: {backtest_results.get('initial_balance', 0):,.0f}")
            print(f"  최종 자본: {backtest_results.get('final_balance', 0):,.2f}")
            print(f"  총 수익률: {backtest_results.get('total_return', 0):.2%}")
            print(f"  총 거래 횟수: {backtest_results.get('total_trades', 0)}")
            print(f"  승률: {backtest_results.get('win_rate', 0):.2%}")
            print(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
    except Exception as e:
        print(f"[추론] 백테스트 실행 실패 (선택사항): {e}")
    
    print("\n" + "=" * 60)
    print("[추론] 추론 완료")
    print("=" * 60)
    
    return {
        'latest_prediction': latest_signal,
        'recent_predictions': recent_results,
        'backtest_results': backtest_results
    }


if __name__ == "__main__":
    main()

