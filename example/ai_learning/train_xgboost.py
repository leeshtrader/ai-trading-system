"""
XGBoost 학습 및 임계값 최적화 스크립트
XGBoost 모델을 학습하고, 학습 후 자동으로 최적 임계값을 계산합니다.

주의: 본 스크립트는 샘플/데모 목적입니다.
This is NOT investment advice. Use at your own risk.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# matplotlib 한글 폰트 경고 필터링
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')

# 경로 설정
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import load_config, get_training_config, get_labeling_config
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel
from ai_learning.train_sample import load_data, create_labels


def train_xgboost(config):
    """XGBoost 모델 학습"""
    print("=" * 60)
    print("[XGBoost] XGBoost 학습 및 임계값 최적화 시작")
    print("=" * 60)
    
    # 설정 읽기
    training_config = get_training_config(config)
    train_test_split = training_config.get('train_test_split', 0.8)
    validation_split = training_config.get('validation_split', 0.8)
    
    # 결과 저장 디렉토리
    models_dir = os.path.join(example_dir, 'models')
    results_dir = os.path.join(example_dir, 'results', 'xgboost_optimized')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 데이터 로드
    print("\n[XGBoost] 데이터 로드 중...")
    data = load_data(config)
    if data is None:
        return None
    
    # 2. 피처 엔지니어링
    print("\n[XGBoost] 피처 엔지니어링: 기술적 지표 생성")
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    print(f"[XGBoost] 기술적 지표 생성 완료: {len(data_with_indicators)}개 데이터 포인트")
    
    # 3. 피처 준비
    print("\n[XGBoost] 피처 준비 중...")
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    labels = create_labels(features_df, data_with_indicators['close'], config)
    
    # 학습/테스트 분할
    split_idx = int(len(features_df) * train_test_split)
    
    X_train = features_df.iloc[:split_idx].values
    X_test = features_df.iloc[split_idx:].values
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    
    print(f"[XGBoost] 학습 데이터: {len(X_train)}개 샘플")
    print(f"[XGBoost] 테스트 데이터: {len(X_test)}개 샘플")
    
    # 검증 데이터 분할
    val_split_idx = int(len(X_train) * validation_split)
    X_train_split = X_train[:val_split_idx]
    X_val_split = X_train[val_split_idx:]
    y_train_split = y_train[:val_split_idx]
    y_val_split = y_train[val_split_idx:]
    
    print(f"[XGBoost] 학습 세트: {len(X_train_split)}개 샘플")
    print(f"[XGBoost] 검증 세트: {len(X_val_split)}개 샘플")
    
    # 4. XGBoost 학습
    print("\n" + "=" * 60)
    print("[XGBoost] XGBoost 모델 학습 시작")
    print("=" * 60)
    
    xgb_config = config.get('models', {}).get('xgboost', {})
    xgb_model = XGBoostDirectionModel(xgb_config)
    
    xgb_model.train(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # 모델 저장
    model_path = os.path.join(models_dir, 'xgboost_direction_model.pkl')
    xgb_model.save(model_path)
    print(f"\n[XGBoost] 모델 저장 완료: {model_path}")
    
    # 5. 학습 결과 평가
    print("\n" + "=" * 60)
    print("[XGBoost] 학습 결과 평가")
    print("=" * 60)
    
    # 테스트 데이터 평가
    feature_names = list(features_df.columns)
    xgb_model.visualize_results(X_test, y_test, feature_names=feature_names, save_dir=results_dir)
    
    # 6. 임계값 최적화
    print("\n" + "=" * 60)
    print("[XGBoost] 임계값 최적화 시작")
    print("=" * 60)
    
    optimal_thresholds = optimize_thresholds(xgb_model, X_val_split, y_val_split, results_dir)
    
    # 7. 최적화된 임계값으로 테스트 데이터 평가
    print("\n" + "=" * 60)
    print("[XGBoost] 최적화된 임계값으로 테스트 데이터 평가")
    print("=" * 60)
    
    evaluate_with_thresholds(xgb_model, X_test, y_test, optimal_thresholds)
    
    # 8. 결과 요약
    print("\n" + "=" * 60)
    print("[XGBoost] 학습 및 최적화 완료")
    print("=" * 60)
    print(f"\n모델 저장 위치: {model_path}")
    print(f"결과 저장 위치: {results_dir}")
    print(f"\n추천 임계값 설정:")
    print(f"  buy_threshold: {optimal_thresholds['buy_threshold']:.3f}")
    print(f"  sell_threshold: {optimal_thresholds['sell_threshold']:.3f}")
    print(f"  strong_signal_threshold: {optimal_thresholds['strong_signal_threshold']:.3f}")
    
    # 추천 임계값을 파일로 저장
    threshold_file = os.path.join(results_dir, 'recommended_thresholds.txt')
    with open(threshold_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("XGBoost 학습 및 임계값 최적화 결과\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("추천 임계값 설정 (sample_config.yaml에 적용):\n\n")
        f.write("ensemble:\n")
        f.write(f"  buy_threshold: {optimal_thresholds['buy_threshold']:.3f}\n")
        f.write(f"  sell_threshold: {optimal_thresholds['sell_threshold']:.3f}\n")
        f.write(f"  strong_signal_threshold: {optimal_thresholds['strong_signal_threshold']:.3f}\n\n")
        f.write("분석 근거:\n")
        f.write(f"  - F1-score 최적화 임계값: {optimal_thresholds.get('optimal_threshold', 'N/A'):.3f}\n")
        f.write(f"  - 균형 잡힌 임계값: {optimal_thresholds.get('balanced_threshold', 'N/A'):.3f}\n")
        if 'statistical_buy' in optimal_thresholds:
            f.write(f"  - 통계적 근거 (BUY): {optimal_thresholds['statistical_buy']:.3f}\n")
            f.write(f"  - 통계적 근거 (SELL): {optimal_thresholds['statistical_sell']:.3f}\n")
    
    print(f"\n임계값 추천 파일 저장: {threshold_file}")
    
    return xgb_model, optimal_thresholds


def optimize_thresholds(xgb_model: XGBoostDirectionModel, 
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       save_dir: str) -> dict:
    """임계값 최적화"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    print("\n[XGBoost] 예측 확률 분포 분석 중...")
    
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
    
    print("\n[XGBoost] 다양한 임계값 테스트 중...")
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
    
    print(f"\n[XGBoost] 최적 임계값 분석 결과:")
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
    
    # 시각화 저장
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import rcParams
        import platform
        
        # 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: DejaVu/NanumGothic)
        def set_korean_font():
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
                    # 폰트가 실제로 사용 가능한지 테스트
                    fig_test = plt.figure(figsize=(1, 1))
                    ax_test = fig_test.add_subplot(111)
                    ax_test.text(0.5, 0.5, '테스트', fontsize=10)
                    plt.close(fig_test)
                    return name
                except Exception:
                    continue
            # 마지막 대비
            rcParams['font.family'] = 'DejaVu Sans'
            return 'DejaVu Sans'
        
        font_name = set_korean_font()
        rcParams['axes.unicode_minus'] = False
        
        # 한글 폰트 경고 필터링
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1-score vs 임계값
        axes[0, 0].plot(results_df['threshold'], results_df['buy_f1'], 'o-', label='BUY F1', alpha=0.7, markersize=3)
        axes[0, 0].plot(results_df['threshold'], results_df['sell_f1'], 's-', label='SELL F1', alpha=0.7, markersize=3)
        axes[0, 0].plot(results_df['threshold'], results_df['avg_f1'], '^-', label='평균 F1', alpha=0.7, markersize=3, linewidth=2)
        axes[0, 0].axvline(optimal_threshold, color='r', linestyle='--', label=f'최적: {optimal_threshold:.3f}')
        axes[0, 0].set_xlabel('임계값')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_title('F1-Score vs 임계값')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall 곡선
        axes[0, 1].plot(results_df['buy_recall'], results_df['buy_precision'], 'o-', label='BUY', alpha=0.7, markersize=3)
        axes[0, 1].plot(results_df['sell_recall'], results_df['sell_precision'], 's-', label='SELL', alpha=0.7, markersize=3)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall 곡선')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 거래 빈도 vs F1-score
        scatter = axes[1, 0].scatter(results_df['trade_frequency'], results_df['avg_f1'], 
                                    c=results_df['threshold'], cmap='viridis', 
                                    s=50, alpha=0.6, edgecolors='black')
        axes[1, 0].axvline(0.05, color='r', linestyle='--', alpha=0.5, label='최소 거래 빈도 (5%)')
        axes[1, 0].scatter([results_df.loc[optimal_idx, 'trade_frequency']], 
                           [results_df.loc[optimal_idx, 'avg_f1']],
                           color='red', s=200, marker='*', label='최적점', edgecolors='black', linewidth=2)
        axes[1, 0].set_xlabel('거래 빈도')
        axes[1, 0].set_ylabel('평균 F1-Score')
        axes[1, 0].set_title('거래 빈도 vs F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='임계값')
        
        # 4. 확률 분포
        axes[1, 1].hist(hold_probs, bins=50, alpha=0.5, label='HOLD', edgecolor='black')
        axes[1, 1].hist(buy_probs, bins=50, alpha=0.5, label='BUY', edgecolor='black')
        axes[1, 1].hist(sell_probs, bins=50, alpha=0.5, label='SELL', edgecolor='black')
        axes[1, 1].axvline(recommended_buy, color='r', linestyle='--', 
                          label=f'추천 BUY: {recommended_buy:.3f}')
        axes[1, 1].axvline(recommended_sell, color='b', linestyle='--', 
                          label=f'추천 SELL: {recommended_sell:.3f}')
        axes[1, 1].set_xlabel('예측 확률')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('예측 확률 분포')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'threshold_optimization.png'), dpi=150)
        plt.close()
        print(f"[XGBoost] 시각화 저장: {os.path.join(save_dir, 'threshold_optimization.png')}")
    except Exception as e:
        print(f"[XGBoost] 시각화 저장 실패: {e}")
    
    return {
        'buy_threshold': round(recommended_buy, 3),
        'sell_threshold': round(recommended_sell, 3),
        'strong_signal_threshold': round(recommended_strong, 3),
        'optimal_threshold': optimal_threshold,
        'balanced_threshold': balanced_threshold,
        'statistical_buy': buy_statistical,
        'statistical_sell': sell_statistical
    }


def evaluate_with_thresholds(xgb_model: XGBoostDirectionModel,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            thresholds: dict):
    """최적화된 임계값으로 테스트 데이터 평가"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    buy_threshold = thresholds['buy_threshold']
    sell_threshold = thresholds['sell_threshold']
    
    # 예측 확률
    proba = xgb_model.model.predict_proba(X_test)
    hold_probs = proba[:, 0]
    buy_probs = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))
    sell_probs = proba[:, 2] if proba.shape[1] > 2 else np.zeros(len(proba))
    
    # 최적화된 임계값으로 예측
    predictions = np.zeros(len(y_test))
    predictions[(buy_probs > buy_threshold) & (buy_probs > sell_probs) & (buy_probs > hold_probs)] = 1  # BUY
    predictions[(sell_probs > sell_threshold) & (sell_probs > buy_probs) & (sell_probs > hold_probs)] = 2  # SELL
    
    # 분류 리포트
    print("\n[XGBoost] 분류 리포트:")
    print(classification_report(y_test, predictions, 
                              target_names=['HOLD', 'BUY', 'SELL'], 
                              zero_division=0))
    
    # 혼동 행렬
    print("\n[XGBoost] 혼동 행렬:")
    cm = confusion_matrix(y_test, predictions)
    print("        예측")
    print("      HOLD  BUY SELL")
    print(f"HOLD  {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
    print(f"BUY   {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
    print(f"SELL  {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")
    
    # 거래 빈도
    trade_count = np.sum(predictions != 0)
    trade_frequency = trade_count / len(y_test)
    print(f"\n[XGBoost] 거래 빈도: {trade_frequency:.2%} ({trade_count}/{len(y_test)})")


def main():
    """메인 함수"""
    config = load_config()
    train_xgboost(config)


if __name__ == "__main__":
    main()

