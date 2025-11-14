"""
LSTM 학습 스크립트 (변동성 기반)

주의: 본 스크립트는 샘플/데모 목적입니다.
This is NOT investment advice. Use at your own risk.
"""
import sys
import os
import pandas as pd
import numpy as np
import warnings

# matplotlib 한글 폰트 경고 필터링
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 경로 설정
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import load_config, get_data_config, get_training_config, get_labeling_config
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.lstm_price import LSTMPriceModel
from ai_learning.train_sample import create_price_targets  # train_sample.py의 함수 사용

# 한글 폰트 설정
import platform
from matplotlib import rcParams, font_manager
import matplotlib
matplotlib.use('Agg')  # 먼저 백엔드 설정

def set_korean_font():
    """한글 폰트 설정 함수"""
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
            # 폰트가 실제로 존재하는지 확인
            font_list = [f.name for f in font_manager.fontManager.ttflist]
            if name in font_list or any(name.lower() in f.lower() for f in font_list):
                rcParams['font.family'] = name
                rcParams['axes.unicode_minus'] = False
                return name
        except Exception:
            continue
    
    # 마지막 대비
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

set_korean_font()

def load_data(config):
    """데이터 로드"""
    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_config = get_data_config(config)
    symbol = data_config.get('symbol', 'NVDA')
    output_dir = data_config.get('output_dir', 'data')
    output_file = data_config.get('output_file', f'{symbol.lower()}_data.csv')
    start_date = data_config.get('start_date', '2019-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    
    data_path = os.path.join(example_dir, output_dir, output_file)
    
    today = datetime.now().date()
    
    if not os.path.exists(data_path):
        print(f"[비교] 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("[비교] 데이터를 다운로드합니다...")
        from data.download_data import download_stock_data, save_data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data = download_stock_data(symbol, start_date, end_date)
        save_data(data, data_path)
        print(f"[비교] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
    else:
        existing_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        last_date = existing_data.index[-1].date()
        
        if last_date < today:
            print(f"[비교] 기존 데이터의 마지막 날짜: {last_date}")
            print(f"[비교] 오늘 날짜: {today}")
            print(f"[비교] 최신 데이터를 다운로드합니다...")
            from data.download_data import download_stock_data, save_data
            data = download_stock_data(symbol, start_date, end_date)
            save_data(data, data_path)
            print(f"[비교] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
        else:
            print(f"[비교] 기존 데이터 사용 (마지막 날짜: {last_date})")
            data = existing_data
    
    print(f"[비교] 데이터 로드 완료: {len(data)}개 데이터 포인트")
    print(f"[비교] 데이터 기간: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data

# create_price_targets_volatility 함수 제거: train_sample.py의 create_price_targets를 사용

def train_lstm_model(sequences_train, targets_train, sequences_val, targets_val, 
                     lstm_config, model_name, save_dir):
    """LSTM 모델 학습"""
    print(f"\n[비교] {model_name} LSTM 학습 시작")
    
    lstm_model = LSTMPriceModel(lstm_config)
    input_size = lstm_config.get('input_size', 5)
    lstm_model.build_model(input_size=input_size)
    
    epochs = lstm_config.get('epochs', 30)
    lstm_model.train(sequences_train, targets_train, 
                     sequences_val=sequences_val, 
                     targets_val=targets_val,
                     epochs=epochs)
    
    # 모델 저장
    model_path = os.path.join(save_dir, f'{model_name}_lstm_model.pth')
    lstm_model.save(model_path)
    print(f"[비교] {model_name} 모델 저장 완료: {model_path}")
    
    return lstm_model

def compare_predictions(models_dict, sequences_test, targets_dict, 
                        data_with_indicators, save_dir):
    """두 모델의 예측 결과 비교"""
    print("\n[비교] 예측 결과 비교 중...")
    
    results = {}
    
    for model_name, model in models_dict.items():
        model.model.eval()
        import torch
        with torch.no_grad():
            X_test = torch.FloatTensor(sequences_test).to(model.device)
            predictions = model.model(X_test).cpu().numpy()
        
        # train_sample.py와 동일한 타겟 사용
        target_key = 'LSTM'
        
        results[model_name] = {
            'predictions': predictions,
            'targets': targets_dict[target_key]
        }
    
    # 통계 비교 (train_sample.py와 동일: 실제 가격 기반)
    print("\n" + "=" * 60)
    print("[비교] 통계 비교 결과 (실제 가격 기반, train_sample.py와 동일)")
    print("=" * 60)
    
    for model_name in results.keys():
        pred = results[model_name]['predictions']
        targ = results[model_name]['targets']
        
        print(f"\n[{model_name}]")
        print(f"  Target Price:")
        print(f"    MAE: {mean_absolute_error(targ[:, 0], pred[:, 0]):.4f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(targ[:, 0], pred[:, 0])):.4f}")
        
        # R² 계산 (실제값의 분산이 0이 아닐 때만)
        targ_var = np.var(targ[:, 0])
        if targ_var > 1e-10:
            r2 = r2_score(targ[:, 0], pred[:, 0])
            print(f"    R²: {r2:.4f}")
        else:
            print(f"    R²: N/A (실제값이 상수)")
        
        # 상관계수
        corr = np.corrcoef(targ[:, 0], pred[:, 0])[0, 1]
        if not np.isnan(corr):
            print(f"    상관계수: {corr:.4f}")
        else:
            print(f"    상관계수: N/A")
        
        # 상대 오차 (MAPE)
        mape = np.mean(np.abs((targ[:, 0] - pred[:, 0]) / (targ[:, 0] + 1e-8))) * 100
        print(f"    MAPE: {mape:.2f}%")
        
        # 범위 겹침 비율
        targ_min, targ_max = np.min(targ[:, 0]), np.max(targ[:, 0])
        in_range_ratio = np.mean((pred[:, 0] >= targ_min) & (pred[:, 0] <= targ_max)) * 100
        print(f"    범위 내 예측 비율: {in_range_ratio:.1f}%")
        
        print(f"    평균 예측값: {np.mean(pred[:, 0]):.2f}, 평균 실제값: {np.mean(targ[:, 0]):.2f}")
        print(f"    예측 범위: [{np.min(pred[:, 0]):.2f}, {np.max(pred[:, 0]):.2f}]")
        print(f"    실제 범위: [{np.min(targ[:, 0]):.2f}, {np.max(targ[:, 0]):.2f}]")
        
        print(f"  Stop Loss:")
        print(f"    MAE: {mean_absolute_error(targ[:, 1], pred[:, 1]):.4f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(targ[:, 1], pred[:, 1])):.4f}")
        
        # R² 계산
        targ_var = np.var(targ[:, 1])
        if targ_var > 1e-10:
            r2 = r2_score(targ[:, 1], pred[:, 1])
            print(f"    R²: {r2:.4f}")
        else:
            print(f"    R²: N/A (실제값이 상수)")
        
        # 상관계수
        corr = np.corrcoef(targ[:, 1], pred[:, 1])[0, 1]
        if not np.isnan(corr):
            print(f"    상관계수: {corr:.4f}")
        else:
            print(f"    상관계수: N/A")
        
        # 상대 오차 (MAPE)
        mape = np.mean(np.abs((targ[:, 1] - pred[:, 1]) / (targ[:, 1] + 1e-8))) * 100
        print(f"    MAPE: {mape:.2f}%")
        
        # 범위 겹침 비율
        targ_min, targ_max = np.min(targ[:, 1]), np.max(targ[:, 1])
        in_range_ratio = np.mean((pred[:, 1] >= targ_min) & (pred[:, 1] <= targ_max)) * 100
        print(f"    범위 내 예측 비율: {in_range_ratio:.1f}%")
        
        print(f"    평균 예측값: {np.mean(pred[:, 1]):.2f}, 평균 실제값: {np.mean(targ[:, 1]):.2f}")
        print(f"    예측 범위: [{np.min(pred[:, 1]):.2f}, {np.max(pred[:, 1]):.2f}]")
        print(f"    실제 범위: [{np.min(targ[:, 1]):.2f}, {np.max(targ[:, 1]):.2f}]")
    
    # 시각화
    visualize_comparison(results, data_with_indicators, save_dir)
    
    return results

def visualize_comparison(results, data_with_indicators, save_dir):
    """결과 시각화"""
    print("\n[비교] 시각화 생성 중...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = list(results.keys())[0]
    result = results[model_name]
    pred = result['predictions']
    targ = result['targets']
    
    # 1. 예측값 vs 실제값 비교
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Target Price (train_sample.py와 동일: 실제 가격)
    axes[0].scatter(targ[:, 0], pred[:, 0], alpha=0.5, s=10)
    min_val = min(targ[:, 0].min(), pred[:, 0].min())
    max_val = max(targ[:, 0].max(), pred[:, 0].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Target Price')
    axes[0].set_ylabel('Predicted Target Price')
    axes[0].set_title(f'{model_name} - Target Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stop Loss (train_sample.py와 동일: 실제 가격)
    axes[1].scatter(targ[:, 1], pred[:, 1], alpha=0.5, s=10)
    min_val = min(targ[:, 1].min(), pred[:, 1].min())
    max_val = max(targ[:, 1].max(), pred[:, 1].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Stop Loss')
    axes[1].set_ylabel('Predicted Stop Loss')
    axes[1].set_title(f'{model_name} - Stop Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=150)
    print(f"[비교] 예측 비교 저장: {os.path.join(save_dir, 'prediction_comparison.png')}")
    plt.close()
    
    # 2. 시계열 비교 (샘플 100개)
    sample_size = min(100, len(targ))
    sample_indices = np.linspace(0, len(targ)-1, sample_size, dtype=int)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Target Price 시계열 (train_sample.py와 동일: 실제 가격)
    axes[0].plot(sample_indices, targ[sample_indices, 0], 
                'o-', label=f'{model_name} Actual', alpha=0.7, markersize=3)
    axes[0].plot(sample_indices, pred[sample_indices, 0], 
                's-', label=f'{model_name} Predicted', alpha=0.7, markersize=3)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Target Price')
    axes[0].set_title('Target Price: Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stop Loss 시계열 (train_sample.py와 동일: 실제 가격)
    axes[1].plot(sample_indices, targ[sample_indices, 1], 
                'o-', label=f'{model_name} Actual', alpha=0.7, markersize=3)
    axes[1].plot(sample_indices, pred[sample_indices, 1], 
                's-', label=f'{model_name} Predicted', alpha=0.7, markersize=3)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Stop Loss')
    axes[1].set_title('Stop Loss: Time Series')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_comparison.png'), dpi=150)
    print(f"[비교] 시계열 비교 저장: {os.path.join(save_dir, 'time_series_comparison.png')}")
    plt.close()
    
    # 3. 오차 분포
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Target Price 오차 (train_sample.py와 동일: 실제 가격)
    errors_target = pred[:, 0] - targ[:, 0]
    axes[0].hist(errors_target, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero Error')
    axes[0].set_xlabel('Prediction Error (Target Price)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{model_name} - Target Price Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stop Loss 오차 (train_sample.py와 동일: 실제 가격)
    errors_stop = pred[:, 1] - targ[:, 1]
    axes[1].hist(errors_stop, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
    axes[1].set_xlabel('Prediction Error (Stop Loss)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name} - Stop Loss Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution_comparison.png'), dpi=150)
    print(f"[비교] 오차 분포 비교 저장: {os.path.join(save_dir, 'error_distribution_comparison.png')}")
    plt.close()
    
    # 4. 목표가/손절가 분포 (train_sample.py와 동일: 실제 가격)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Target Price 분포 (train_sample.py와 동일: 실제 가격)
    axes[0].hist(targ[:, 0], bins=30, alpha=0.5, label=f'{model_name} Actual', edgecolor='black')
    axes[0].hist(pred[:, 0], bins=30, alpha=0.3, label=f'{model_name} Predicted', edgecolor='black', linestyle='--')
    axes[0].set_xlabel('Target Price')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Price Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stop Loss 분포 (train_sample.py와 동일: 실제 가격)
    axes[1].hist(targ[:, 1], bins=30, alpha=0.5, label=f'{model_name} Actual', edgecolor='black')
    axes[1].hist(pred[:, 1], bins=30, alpha=0.3, label=f'{model_name} Predicted', edgecolor='black', linestyle='--')
    axes[1].set_xlabel('Stop Loss')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Stop Loss Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'price_distribution_comparison.png'), dpi=150)
    print(f"[비교] 가격 분포 비교 저장: {os.path.join(save_dir, 'price_distribution_comparison.png')}")
    plt.close()
    
    print("[비교] 시각화 완료")

def main():
    """메인 함수"""
    print("=" * 60)
    print("[LSTM] LSTM 학습 시작 (train_sample.py와 동일한 방식)")
    print("=" * 60)
    
    # 설정 로드
    config = load_config()
    
    # 결과 저장 디렉토리
    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(example_dir, 'results', 'lstm_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 데이터 로드
    data = load_data(config)
    if data is None:
        return
    
    # 2. 피처 엔지니어링
    print("\n[비교] 피처 엔지니어링: 기술적 지표 생성")
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    print(f"[비교] 기술적 지표 생성 완료: {len(data_with_indicators)}개 데이터 포인트")
    
    # 3. LSTM 시퀀스 준비
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    sequences = feature_eng.prepare_sequences_for_lstm(data_with_indicators, sequence_length=sequence_length)
    
    # 4. 학습/테스트 분할 (train_sample.py와 동일한 방식)
    training_config = get_training_config(config)
    train_test_split = training_config.get('train_test_split', 0.8)
    validation_split = training_config.get('validation_split', 0.8)
    
    # train_sample.py와 동일: features_df 기준으로 분할
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    split_idx = int(len(features_df) * train_test_split)
    
    # 시퀀스는 sequence_length만큼 앞서 시작하므로 분할도 동일하게
    seq_train = sequences[:split_idx-sequence_length]
    seq_test = sequences[split_idx-sequence_length:]
    
    # 검증 데이터 분할
    val_split_idx = int(len(seq_train) * validation_split)
    
    # 5. 가격 목표 생성 (train_sample.py의 create_price_targets 사용)
    print("\n[비교] 가격 목표 생성 (train_sample.py와 동일한 방식)")
    price_targets = create_price_targets(data_with_indicators['close'], data_with_indicators, config)
    # create_price_targets는 이미 sequence_length부터 시작하므로 추가 슬라이싱 불필요
    
    target_train = price_targets[:split_idx-sequence_length]
    target_test = price_targets[split_idx-sequence_length:]
    
    # 검증 데이터 분할
    seq_train_split = seq_train[:val_split_idx]
    seq_val_split = seq_train[val_split_idx:]
    target_train_split = target_train[:val_split_idx]
    target_val_split = target_train[val_split_idx:]
    
    # 6. LSTM 학습 (train_sample.py와 동일한 방식)
    model_lstm = train_lstm_model(
        seq_train_split, target_train_split,
        seq_val_split, target_val_split,
        lstm_config, 'lstm', save_dir
    )
    
    # 7. 예측 결과 분석
    models_dict = {
        'LSTM': model_lstm
    }
    
    # 테스트 데이터에 맞게 targets 조정
    target_test_aligned = target_test[:len(seq_test)]
    
    # 타겟 사용
    results = {}
    results['LSTM'] = {
        'predictions': None,
        'targets': target_test_aligned
    }
    
    # 예측 수행
    import torch
    for model_name, model in models_dict.items():
        model.model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(seq_test).to(model.device)
            predictions = model.model(X_test).cpu().numpy()
        results[model_name]['predictions'] = predictions
    
    # 분석 및 시각화
    targets_dict = {
        'LSTM': target_test_aligned
    }
    compare_predictions(models_dict, seq_test, targets_dict,
                       data_with_indicators, save_dir)
    
    print("\n" + "=" * 60)
    print("[LSTM] LSTM 학습 완료 (train_sample.py와 동일한 방식)")
    print(f"[LSTM] 결과 저장 위치: {save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

