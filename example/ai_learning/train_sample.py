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
import yaml
from datetime import datetime

# 경로 설정
# train_sample.py는 example/ai_learning/에 있으므로
# example_dir는 두 단계 위로 올라가야 함
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import get_data_config, get_training_config, get_labeling_config, get_backtest_config, load_config
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel
from ai_learning.models.lstm_price import LSTMPriceModel
from ai_learning.models.rl_agent import TradingRLAgent
from ai_learning.ensemble.role_based_ensemble import RoleBasedEnsemble
from sklearn.metrics import precision_score, recall_score, f1_score

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
    """데이터 로드 (최신 데이터 확인 및 다운로드 포함)"""
    # train_sample.py는 example/ai_learning/에 있으므로
    # example_dir는 두 단계 위로 올라가야 함
    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 설정 파일에서 데이터 경로 읽기
    data_config = get_data_config(config)
    symbol = data_config.get('symbol', 'NVDA')
    output_dir = data_config.get('output_dir', 'data')
    output_file = data_config.get('output_file', f'{symbol.lower()}_data.csv')
    start_date = data_config.get('start_date', '2019-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    
    data_path = os.path.join(example_dir, output_dir, output_file)
    
    # 오늘 날짜 확인
    today = datetime.now().date()
    
    # 데이터 파일이 있는지 확인하고, 최신 데이터인지 확인
    if not os.path.exists(data_path):
        print(f"[샘플] 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("[샘플] 데이터를 다운로드합니다...")
        from data.download_data import download_stock_data, save_data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data = download_stock_data(symbol, start_date, end_date)
        save_data(data, data_path)
        print(f"[샘플] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
    else:
        # 기존 데이터 로드하여 마지막 날짜 확인
        existing_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # 빈 데이터프레임 체크
        if len(existing_data) == 0:
            print(f"[샘플] 기존 데이터 파일이 비어있습니다: {data_path}")
            print("[샘플] 데이터를 다운로드합니다...")
            from data.download_data import download_stock_data, save_data
            data = download_stock_data(symbol, start_date, end_date)
            save_data(data, data_path)
            print(f"[샘플] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
        else:
            last_date = existing_data.index[-1].date()
            
            # 마지막 날짜가 오늘보다 오래되었으면 최신 데이터 다운로드
            if last_date < today:
                print(f"[샘플] 기존 데이터의 마지막 날짜: {last_date}")
                print(f"[샘플] 오늘 날짜: {today}")
                print(f"[샘플] 최신 데이터를 다운로드합니다...")
                from data.download_data import download_stock_data, save_data
                data = download_stock_data(symbol, start_date, end_date)
                save_data(data, data_path)
                print(f"[샘플] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
            else:
                print(f"[샘플] 기존 데이터 사용 (마지막 날짜: {last_date})")
                data = existing_data
    
    print(f"[샘플] 데이터 로드 완료: {len(data)}개 데이터 포인트 (경로: {data_path})")
    
    # 데이터가 비어있지 않은지 확인
    if len(data) == 0:
        print("[샘플] 경고: 로드된 데이터가 비어있습니다.")
        return None
    
    print(f"[샘플] 데이터 기간: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data

def create_labels(features_df: pd.DataFrame, price_data: pd.Series, config) -> np.ndarray:
    """
    개선된 레이블 생성 (시간적 일관성 개선)
    - 다음 날 가격 변화를 기준으로 레이블 생성
    - Rolling window 기반 분위수 계산으로 시장 환경 변화에 적응
    - 변동성 기반 필터링으로 노이즈 제거
    - 기술적 지표 이중 사용 최소화 (순환 참조 방지)
    """
    labeling_config = get_labeling_config(config)
    buy_percentile = labeling_config.get('buy_percentile', 70)
    sell_percentile = labeling_config.get('sell_percentile', 30)
    
    # 최소 가격 변화 임계값 (노이즈 필터링)
    min_price_change = labeling_config.get('min_price_change_threshold', 0.008)
    
    # Rolling window 크기 (기본값: 252일 = 1년)
    percentile_window_size = labeling_config.get('percentile_window_size', 252)
    
    # 변동성 필터링 배수 (설정 파일에서 읽기, 기본값 사용)
    buy_volatility_multiplier = labeling_config.get('buy_volatility_multiplier', 2.0)
    sell_volatility_multiplier = labeling_config.get('sell_volatility_multiplier', 3.0)
    
    labels = []
    price_changes = []
    
    # 변동성 계산 (노이즈 필터링용)
    volatility_window = labeling_config.get('volatility_window', 60)
    price_changes_short = price_data.pct_change()
    volatilities = price_changes_short.rolling(window=volatility_window).std()
    
    # 다음 날 가격 변화 계산
    for i in range(len(features_df) - 1):
        current_price = price_data.iloc[i]
        next_price = price_data.iloc[i + 1]
        price_change = (next_price - current_price) / current_price
        price_changes.append(price_change)
    
    # Rolling window 기반 분위수 계산 및 레이블 생성
    if len(price_changes) > 0:
        print(f"[샘플] 레이블 생성: Rolling window 기반 분위수 계산 (윈도우 크기: {percentile_window_size}일)")
        
        for idx, price_change in enumerate(price_changes):
            i = idx  # 원본 인덱스
            
            # Rolling window 기반 분위수 계산
            if idx < percentile_window_size:
                # 초기에는 사용 가능한 데이터만 사용
                window_changes = price_changes[:idx+1]
            else:
                # 최근 N일의 데이터만 사용 (시간적 일관성 확보)
                window_changes = price_changes[idx-percentile_window_size+1:idx+1]
            
            # 현재 윈도우의 분위수 계산
            if len(window_changes) > 0:
                window_changes_array = np.array(window_changes)
                buy_threshold = np.percentile(window_changes_array, buy_percentile)
                sell_threshold = np.percentile(window_changes_array, sell_percentile)
            else:
                # 윈도우 데이터가 없으면 기본값 사용
                buy_threshold = 0.01
                sell_threshold = -0.01
            
            # 현재 변동성
            current_volatility = volatilities.iloc[i] if i < len(volatilities) and not np.isnan(volatilities.iloc[i]) else 0.02
            
            # 변동성 기반 최소 변화 임계값
            volatility_adjusted_threshold = min_price_change * (1 + current_volatility)
            
            # 레이블 생성 로직 (기술적 지표 제거, 순환 참조 방지)
            is_buy_signal = False
            is_sell_signal = False
            
            # 분위수 기준 확인 (기술적 지표 없이 순수 분위수 기반)
            if price_change >= buy_threshold:
                # 분위수 기준 충족 + 변동성 필터링
                if abs(price_change) >= volatility_adjusted_threshold * buy_volatility_multiplier:
                    is_buy_signal = True
            
            elif price_change <= sell_threshold:
                # 분위수 기준 충족 + 변동성 필터링 (SELL은 더 엄격)
                if abs(price_change) >= volatility_adjusted_threshold * sell_volatility_multiplier:
                    is_sell_signal = True
            
            # 최종 레이블 결정
            if is_buy_signal and abs(price_change) >= volatility_adjusted_threshold:
                labels.append(1)  # BUY
            elif is_sell_signal and abs(price_change) >= volatility_adjusted_threshold:
                labels.append(2)  # SELL
            else:
                labels.append(0)  # HOLD
    
    # 마지막 데이터는 레이블이 없으므로 HOLD
    labels.append(0)
    
    return np.array(labels)

def create_price_targets(price_data: pd.Series, data_with_indicators: pd.DataFrame, config) -> np.ndarray:
    """
    가격 목표 생성 (변동성 기반)
    
    Args:
        price_data: 가격 시리즈
        data_with_indicators: 기술적 지표가 포함된 데이터프레임 (변동성 포함)
        config: 설정 딕셔너리
    """
    labeling_config = get_labeling_config(config)
    time_horizon = labeling_config.get('default_time_horizon', 5)
    confidence = labeling_config.get('default_confidence', 0.7)
    
    targets = []
    
    # 변동성 기반 계산
    volatility_multiplier = labeling_config.get('volatility_multiplier', 2.0)
    volatility_window = labeling_config.get('volatility_window', 60)
    consider_transaction_cost = labeling_config.get('consider_transaction_cost', True)
    
    # 수수료 고려한 최소/최대 기준 계산
    if consider_transaction_cost:
        # 백테스트 설정에서 수수료율 읽기
        backtest_config = config.get('backtest', {})
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
    
    # 변동성 계산 (롤링 표준편차)
    if 'volatility' in data_with_indicators.columns:
        volatilities = data_with_indicators['volatility'].values
    else:
        # 변동성이 없으면 계산
        price_changes = price_data.pct_change()
        volatilities = price_changes.rolling(window=volatility_window).std().values
    
    # 트렌드 기반 조정 설정
    use_trend_adjustment = labeling_config.get('use_trend_adjustment', True)
    trend_ma_window = labeling_config.get('trend_ma_window', 20)
    trend_strength_threshold = labeling_config.get('trend_strength_threshold', 0.005)
    trend_adjustment_factor = labeling_config.get('trend_adjustment_factor', 0.3)
    
    # 이동평균선 계산 (트렌드 판단용)
    if use_trend_adjustment:
        # MA 컬럼이 있으면 사용, 없으면 계산
        ma_column = f'ma_{trend_ma_window}'
        if ma_column in data_with_indicators.columns:
            ma_values = data_with_indicators[ma_column].values
        else:
            ma_values = price_data.rolling(window=trend_ma_window).mean().values
    else:
        ma_values = None
    
    # sequences와 크기를 맞추기 위해 sequence_length만큼 앞서 시작
    # sequences는 [sequence_length:] 인덱스부터 시작하므로, targets도 동일하게 맞춤
    # 하지만 실제로는 각 시퀀스의 마지막 시점에 대한 타겟이 필요하므로,
    # sequence_length 인덱스부터 시작하여 각 시퀀스의 마지막 시점에 대한 타겟 생성
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    
    # sequence_length 인덱스부터 시작하여 각 시퀀스의 마지막 시점에 대한 타겟 생성
    for i in range(sequence_length, len(price_data)):
        current_price = price_data.iloc[i]
        # 다음 가격이 있으면 사용, 없으면 현재 가격 사용
        if i < len(price_data) - 1:
            next_price = price_data.iloc[i + 1]
        else:
            next_price = current_price
        
        # 변동성 기반 목표가/손절가 계산 (기본값)
        if i < len(volatilities) and not np.isnan(volatilities[i]) and volatilities[i] > 0:
            vol = volatilities[i]
            # 변동성의 N배를 목표/손절로 사용
            target_ratio = 1.0 + (vol * volatility_multiplier)
            stop_loss_ratio = 1.0 - (vol * volatility_multiplier)
        else:
            # 변동성 데이터가 없으면 최소/최대 범위의 중간값 사용
            target_ratio = (min_target_ratio + max_target_ratio) / 2.0
            stop_loss_ratio = (min_stop_loss_ratio + max_stop_loss_ratio) / 2.0
        
        # 트렌드 기반 조정 (변동성 기반 계산 후)
        if use_trend_adjustment and ma_values is not None and i < len(ma_values):
            ma_value = ma_values[i]
            if not np.isnan(ma_value) and ma_value > 0:
                # 트렌드 방향 판단
                price_ma_ratio = current_price / ma_value
                
                # 트렌드 조정량 계산 (변동성 기반)
                if i < len(volatilities) and not np.isnan(volatilities[i]) and volatilities[i] > 0:
                    vol = volatilities[i]
                    trend_adjustment = vol * volatility_multiplier * trend_adjustment_factor
                else:
                    # 변동성 데이터가 없으면 기본값 사용
                    trend_adjustment = 0.01 * volatility_multiplier * trend_adjustment_factor
                
                # 상승 트렌드: 현재가 > MA × (1 + threshold)
                if price_ma_ratio > (1.0 + trend_strength_threshold):
                    # 목표가 상향 조정, 손절가 상향 조정 (덜 낮게)
                    target_ratio += trend_adjustment
                    stop_loss_ratio += trend_adjustment * 0.5  # 손절가는 덜 조정
                
                # 하락 트렌드: 현재가 < MA × (1 - threshold)
                elif price_ma_ratio < (1.0 - trend_strength_threshold):
                    # 목표가 하향 조정 (덜 높게), 손절가 하향 조정
                    target_ratio -= trend_adjustment * 0.5  # 목표가는 덜 조정
                    stop_loss_ratio -= trend_adjustment  # 손절가는 더 조정
                
                # 횡보: 기존 변동성 기반 계산 유지 (조정 없음)
        
        # 최소/최대 비율로 제한
        target_ratio = np.clip(target_ratio, min_target_ratio, max_target_ratio)
        stop_loss_ratio = np.clip(stop_loss_ratio, min_stop_loss_ratio, max_stop_loss_ratio)
        
        # 목표가/손절가 계산 (현재 가격 기준)
        target_price = current_price * target_ratio
        stop_loss = current_price * stop_loss_ratio
        
        # 실제 변동성 (다음 가격 변화율)
        if i < len(price_data) - 1:
            volatility = abs((next_price - current_price) / current_price)
        else:
            volatility = 0.01  # 마지막 데이터는 기본값 사용
        
        targets.append([target_price, stop_loss, time_horizon, confidence, volatility])
    
    return np.array(targets)

def optimize_thresholds(xgb_model: XGBoostDirectionModel, 
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       save_dir: str) -> dict:
    """임계값 최적화 (tools/optimize_thresholds.py와 동일한 로직)"""
    print("\n[샘플] 예측 확률 분포 분석 중...")
    
    # 예측 확률 계산
    proba = xgb_model.model.predict_proba(X_val)
    
    # 클래스별 확률 추출 (0=HOLD, 1=BUY, 2=SELL)
    hold_probs = proba[:, 0]
    buy_probs = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))
    sell_probs = proba[:, 2] if proba.shape[1] > 2 else np.zeros(len(proba))
    
    # 통계 정보
    buy_mean = np.mean(buy_probs)
    buy_std = np.std(buy_probs)
    sell_mean = np.mean(sell_probs)
    sell_std = np.std(sell_probs)
    
    print(f"  BUY 확률: 평균={buy_mean:.3f}, 표준편차={buy_std:.3f}")
    print(f"  SELL 확률: 평균={sell_mean:.3f}, 표준편차={sell_std:.3f}")
    
    # 임계값 범위 (0.3 ~ 0.9, 0.01 간격)
    thresholds = np.arange(0.3, 0.91, 0.01)
    
    results = []
    
    print("\n[샘플] 다양한 임계값 테스트 중...")
    for threshold in thresholds:
        # BUY 예측
        buy_pred = (buy_probs > threshold) & (buy_probs > sell_probs) & (buy_probs > hold_probs)
        
        # SELL 예측
        sell_pred = (sell_probs > threshold) & (sell_probs > buy_probs) & (sell_probs > hold_probs)
        
        # 실제 레이블과 비교
        y_buy = (y_val == 1).astype(int)
        y_sell = (y_val == 2).astype(int)
        
        buy_precision = precision_score(y_buy, buy_pred, zero_division=0)
        buy_recall = recall_score(y_buy, buy_pred, zero_division=0)
        buy_f1 = f1_score(y_buy, buy_pred, zero_division=0)
        
        sell_precision = precision_score(y_sell, sell_pred, zero_division=0)
        sell_recall = recall_score(y_sell, sell_pred, zero_division=0)
        sell_f1 = f1_score(y_sell, sell_pred, zero_division=0)
        
        # 평균 F1-score
        avg_f1 = (buy_f1 + sell_f1) / 2.0
        
        # 거래 빈도
        trade_frequency = (np.sum(buy_pred) + np.sum(sell_pred)) / len(y_val)
        
        results.append({
            'threshold': threshold,
            'buy_precision': buy_precision,
            'buy_recall': buy_recall,
            'buy_f1': buy_f1,
            'sell_precision': sell_precision,
            'sell_recall': sell_recall,
            'sell_f1': sell_f1,
            'avg_f1': avg_f1,
            'trade_frequency': trade_frequency
        })
    
    results_df = pd.DataFrame(results)
    
    # 최적 임계값 찾기 (F1-score 최대화)
    optimal_idx = results_df['avg_f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    # 거래 빈도가 너무 낮은 경우 고려
    filtered_results = results_df[
        (results_df['trade_frequency'] >= 0.05) &  # 최소 5% 거래 빈도
        (results_df['avg_f1'] >= results_df['avg_f1'].max() * 0.9)  # 최대 F1의 90% 이상
    ]
    
    if len(filtered_results) > 0:
        # 거래 빈도와 F1-score의 균형
        filtered_results = filtered_results.copy()
        filtered_results['balanced_score'] = (
            filtered_results['avg_f1'] * 0.7 + 
            filtered_results['trade_frequency'] * 0.3
        )
        balanced_idx = filtered_results['balanced_score'].idxmax()
        balanced_threshold = filtered_results.loc[balanced_idx, 'threshold']
    else:
        balanced_threshold = optimal_threshold
    
    print(f"\n[샘플] 최적 임계값 분석 결과:")
    print(f"  F1-score 최대화 임계값: {optimal_threshold:.3f}")
    print(f"    - 평균 F1: {results_df.loc[optimal_idx, 'avg_f1']:.3f}")
    print(f"    - 거래 빈도: {results_df.loc[optimal_idx, 'trade_frequency']:.2%}")
    
    if balanced_threshold != optimal_threshold:
        balanced_idx = filtered_results['balanced_score'].idxmax()
        print(f"\n  균형 잡힌 임계값: {balanced_threshold:.3f}")
        print(f"    - 평균 F1: {filtered_results.loc[balanced_idx, 'avg_f1']:.3f}")
        print(f"    - 거래 빈도: {filtered_results.loc[balanced_idx, 'trade_frequency']:.2%}")
    
    # 통계적 근거
    buy_statistical = max(0.5, buy_mean + 1.5 * buy_std)
    sell_statistical = max(0.5, sell_mean + 1.5 * sell_std)
    
    # 최종 추천: F1 최적화와 통계적 근거의 평균
    recommended_buy = np.clip((optimal_threshold + balanced_threshold + buy_statistical) / 3.0, 0.4, 0.8)
    recommended_sell = np.clip((optimal_threshold + balanced_threshold + sell_statistical) / 3.0, 0.4, 0.8)
    recommended_strong = min(0.8, recommended_buy + 0.1)
    
    # 결과 딕셔너리 반환
    optimal_thresholds = {
        'optimal_threshold': optimal_threshold,
        'balanced_threshold': balanced_threshold,
        'buy_threshold': recommended_buy,
        'sell_threshold': recommended_sell,
        'strong_signal_threshold': recommended_strong,
        'statistical_buy': buy_statistical,
        'statistical_sell': sell_statistical
    }
    
    # 결과를 파일로 저장
    threshold_file = os.path.join(save_dir, 'recommended_thresholds.txt')
    os.makedirs(save_dir, exist_ok=True)
    with open(threshold_file, 'w', encoding='utf-8') as f:
        f.write("추천 임계값 설정:\n")
        f.write(f"  buy_threshold: {recommended_buy:.3f}\n")
        f.write(f"  sell_threshold: {recommended_sell:.3f}\n")
        f.write(f"  strong_signal_threshold: {recommended_strong:.3f}\n")
        f.write(f"\n분석 근거:\n")
        f.write(f"  - F1-score 최적화 임계값: {optimal_threshold:.3f}\n")
        f.write(f"  - 균형 잡힌 임계값: {balanced_threshold:.3f}\n")
        f.write(f"  - 통계적 근거 (BUY): {buy_statistical:.3f}\n")
        f.write(f"  - 통계적 근거 (SELL): {sell_statistical:.3f}\n")
    
    print(f"\n[샘플] 임계값 추천 파일 저장: {threshold_file}")
    
    return optimal_thresholds

def update_config_thresholds(optimal_thresholds: dict, config_path: str = None):
    """
    최적화된 임계값을 config 파일에 자동으로 업데이트
    
    Args:
        optimal_thresholds: 최적화된 임계값 딕셔너리
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
    """
    if config_path is None:
        # 기본 경로: example/config/sample_config.yaml
        config_path = os.path.join(example_dir, 'config', 'sample_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"[샘플] 경고: 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    
    # YAML 파일 읽기
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ensemble 섹션 찾기 및 업데이트
    ensemble_start_idx = None
    ensemble_end_idx = None
    in_ensemble = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # ensemble 섹션 시작 찾기
        if stripped.startswith('ensemble:') and not in_ensemble:
            ensemble_start_idx = i
            in_ensemble = True
            indent_level = len(line) - len(line.lstrip())
            continue
        
        if in_ensemble:
            # 다음 최상위 섹션 시작 시 종료
            current_indent = len(line) - len(line.lstrip())
            if stripped and not stripped.startswith('#') and current_indent <= indent_level and i > ensemble_start_idx:
                ensemble_end_idx = i
                break
    
    # ensemble 섹션이 끝나지 않았으면 파일 끝까지
    if in_ensemble and ensemble_end_idx is None:
        ensemble_end_idx = len(lines)
    
    # 임계값 업데이트
    buy_threshold = optimal_thresholds.get('buy_threshold', 0.35)
    sell_threshold = optimal_thresholds.get('sell_threshold', 0.35)
    strong_signal_threshold = optimal_thresholds.get('strong_signal_threshold', 0.4)
    
    if ensemble_start_idx is not None:
        # 기존 ensemble 섹션 내에서 임계값 찾아서 업데이트
        updated_lines = []
        threshold_updated = {'buy': False, 'sell': False, 'strong': False}
        
        for i, line in enumerate(lines):
            if i < ensemble_start_idx or (ensemble_end_idx is not None and i >= ensemble_end_idx):
                # ensemble 섹션 밖이면 그대로 추가
                updated_lines.append(line)
            else:
                # ensemble 섹션 내
                stripped = line.strip()
                # 주석 처리된 라인도 포함하여 확인
                if 'buy_threshold' in stripped and (stripped.startswith('buy_threshold:') or stripped.startswith('# buy_threshold:')):
                    # 기존 값 업데이트 (주석 해제)
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(f"{' ' * indent}buy_threshold: {buy_threshold:.3f}  # 자동 최적화됨\n")
                    threshold_updated['buy'] = True
                elif 'sell_threshold' in stripped and (stripped.startswith('sell_threshold:') or stripped.startswith('# sell_threshold:')):
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(f"{' ' * indent}sell_threshold: {sell_threshold:.3f}  # 자동 최적화됨\n")
                    threshold_updated['sell'] = True
                elif 'strong_signal_threshold' in stripped and (stripped.startswith('strong_signal_threshold:') or stripped.startswith('# strong_signal_threshold:')):
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(f"{' ' * indent}strong_signal_threshold: {strong_signal_threshold:.3f}  # 자동 최적화됨\n")
                    threshold_updated['strong'] = True
                else:
                    updated_lines.append(line)
        
        # 누락된 임계값 추가 (ensemble 섹션 끝에 추가)
        if not all(threshold_updated.values()):
            # ensemble 섹션의 마지막 줄 찾기
            last_ensemble_line = ensemble_end_idx - 1 if ensemble_end_idx else len(updated_lines) - 1
            while last_ensemble_line >= 0 and (last_ensemble_line >= len(updated_lines) or 
                                                (updated_lines[last_ensemble_line].strip() == '' or 
                                                 updated_lines[last_ensemble_line].strip().startswith('#'))):
                last_ensemble_line -= 1
            
            if last_ensemble_line >= 0:
                indent = len(updated_lines[last_ensemble_line]) - len(updated_lines[last_ensemble_line].lstrip())
                if not threshold_updated['buy']:
                    updated_lines.insert(last_ensemble_line + 1, f"{' ' * indent}buy_threshold: {buy_threshold:.3f}  # 자동 최적화됨\n")
                if not threshold_updated['sell']:
                    updated_lines.insert(last_ensemble_line + 1, f"{' ' * indent}sell_threshold: {sell_threshold:.3f}  # 자동 최적화됨\n")
                if not threshold_updated['strong']:
                    updated_lines.insert(last_ensemble_line + 1, f"{' ' * indent}strong_signal_threshold: {strong_signal_threshold:.3f}  # 자동 최적화됨\n")
        
        lines = updated_lines
    else:
        # ensemble 섹션이 없으면 추가 (파일 끝에)
        lines.append('\n# 앙상블 설정 (자동 최적화됨)\n')
        lines.append('ensemble:\n')
        lines.append(f'  buy_threshold: {buy_threshold:.3f}  # 자동 최적화됨\n')
        lines.append(f'  sell_threshold: {sell_threshold:.3f}  # 자동 최적화됨\n')
        lines.append(f'  strong_signal_threshold: {strong_signal_threshold:.3f}  # 자동 최적화됨\n')
    
    # 파일 쓰기
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\n[샘플] 설정 파일 자동 업데이트 완료: {config_path}")
    print(f"  buy_threshold: {buy_threshold:.3f}")
    print(f"  sell_threshold: {sell_threshold:.3f}")
    print(f"  strong_signal_threshold: {strong_signal_threshold:.3f}")

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
    enable_gpu = training_config.get('enable_gpu', False)
    
    # GPU 사용 여부에 따라 디바이스 설정
    import torch
    if enable_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"[샘플] GPU 사용 활성화: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        if enable_gpu:
            print("[샘플] GPU 사용 요청되었으나 사용 불가능 (CPU 사용)")
        else:
            print("[샘플] CPU 사용 (enable_gpu: false)")
    
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
    price_targets = create_price_targets(data_with_indicators['close'], data_with_indicators, config)
    
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
    # GPU 사용 여부에 따라 tree_method 설정
    if enable_gpu:
        try:
            import xgboost as xgb
            # GPU 사용 가능 여부 확인 (XGBoost GPU 지원 확인)
            xgb_config['tree_method'] = 'gpu_hist'  # GPU 사용
            print("[샘플] XGBoost GPU 사용 활성화 (tree_method: gpu_hist)")
        except:
            print("[샘플] XGBoost GPU 사용 불가능 (CPU 사용)")
            xgb_config['tree_method'] = 'hist'  # CPU 사용
    else:
        xgb_config['tree_method'] = 'hist'  # CPU 사용
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
    
    # 임계값 최적화 (자동 적용)
    print("\n[샘플] XGBoost 임계값 최적화 시작")
    optimal_thresholds = optimize_thresholds(xgb_model, X_val_split, y_val_split, save_dir)
    
    # 최적화된 임계값을 설정 파일에 자동 반영
    config_path = os.path.join(example_dir, 'config', 'sample_config.yaml')
    update_config_thresholds(optimal_thresholds, config_path)
    
    # 메모리의 config 딕셔너리도 업데이트 (현재 학습 세션에서 사용하기 위해)
    if 'ensemble' not in config:
        config['ensemble'] = {}
    config['ensemble']['buy_threshold'] = optimal_thresholds.get('buy_threshold', 0.35)
    config['ensemble']['sell_threshold'] = optimal_thresholds.get('sell_threshold', 0.35)
    config['ensemble']['strong_signal_threshold'] = optimal_thresholds.get('strong_signal_threshold', 0.4)
    print("[샘플] 현재 학습 세션의 config도 업데이트 완료")
    
    # 5. LSTM 학습
    print("\n[샘플] LSTM 학습 시작")
    # GPU 사용 여부에 따라 device 설정
    lstm_config['device'] = device
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
    
    # 5.5. 보상 함수 파라미터 자동 최적화 (옵션)
    training_config = get_training_config(config)
    auto_optimize = training_config.get('auto_optimize_reward_params', False)
    
    if auto_optimize:
        print("\n[샘플] 보상 함수 파라미터 자동 최적화 시작 (Sharpe Ratio 기반)")
        print("[샘플] 주의: 최적화는 시간이 오래 걸릴 수 있습니다.")
        try:
            # 최적화 스크립트 import
            tools_dir = os.path.join(example_dir, 'tools')
            sys.path.insert(0, tools_dir)
            from optimize_reward_params import optimize_reward_parameters, update_config_reward_params
            
            # 최적화 실행
            n_trials = training_config.get('reward_optimization_trials', 10)
            config_path = os.path.join(example_dir, 'config', 'sample_config.yaml')
            
            best_params, optimal_rl_config, _ = optimize_reward_parameters(
                config_path=config_path, 
                n_trials=n_trials
            )
            
            # config 파일 자동 업데이트
            update_config_reward_params(optimal_rl_config, config_path)
            
            # 메모리의 config도 업데이트 (현재 학습 세션에서 사용하기 위해)
            if 'models' not in config:
                config['models'] = {}
            if 'rl' not in config['models']:
                config['models']['rl'] = {}
            config['models']['rl'].update(optimal_rl_config)
            
            print("[샘플] 보상 함수 파라미터 최적화 완료 및 config 파일 업데이트 완료")
            print("[샘플] 최적화된 파라미터로 RL 학습을 진행합니다.")
        except Exception as e:
            print(f"[샘플] 보상 함수 파라미터 최적화 실패: {e}")
            print("[샘플] 기존 파라미터로 RL 학습을 진행합니다.")
            import traceback
            traceback.print_exc()
    
    # 6. 강화학습 학습 (선택사항 - stable-baselines3가 설치된 경우만)
    print("\n[샘플] 강화학습 학습 시작")
    try:
        price_array = data_with_indicators['close'].values
        rl_config = config.get('models', {}).get('rl', {})
        rl_agent = TradingRLAgent(rl_config)
        
        # RL 학습 시 XGBoost/LSTM 모델과 데이터 전달 (학습-예측 환경 일치)
        # 학습 데이터 준비 (시퀀스 길이 고려)
        lstm_config = config.get('models', {}).get('lstm', {})
        sequence_length = lstm_config.get('sequence_length', 20)
        
        # 학습용 피처와 시퀀스 (시퀀스 길이만큼 앞에서 시작)
        train_features = features_df.iloc[:split_idx]
        train_sequences = sequences[:split_idx-sequence_length] if len(sequences) > split_idx-sequence_length else sequences
        
        # 학습용 가격 데이터 (시퀀스 길이만큼 앞에서 시작)
        train_price_array = price_array[:split_idx]
        
        rl_agent.train(
            price_data=train_price_array,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            features_data=train_features,
            sequences_data=train_sequences
        )
        
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
        
        # 10. 모델 저장 (설정에서 활성화된 경우)
        training_config = get_training_config(config)
        save_models = training_config.get('save_models', True)
        if save_models:
            print("\n[샘플] 학습된 모델 저장 중...")
            models_dir = training_config.get('models_dir', 'models')
            models_path = os.path.join(example_dir, models_dir)
            os.makedirs(models_path, exist_ok=True)
            
            # XGBoost 모델 저장
            xgb_path = os.path.join(models_path, 'xgboost_model.pkl')
            xgb_model.save(xgb_path)
            
            # LSTM 모델 저장
            lstm_path = os.path.join(models_path, 'lstm_model.pth')
            lstm_model.save(lstm_path)
            
            # RL Agent 저장
            try:
                rl_path = os.path.join(models_path, 'rl_agent.zip')
                rl_agent.save(rl_path)
            except Exception as e:
                print(f"[샘플] RL Agent 저장 실패 (선택사항): {e}")
            
            print(f"[샘플] 모델 저장 완료: {models_path}")
        
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

