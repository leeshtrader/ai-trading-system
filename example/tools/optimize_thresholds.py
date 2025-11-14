"""
임계값 최적화 스크립트
XGBoost 예측 확률 분포를 분석하여 최적의 buy_threshold, sell_threshold를 계산합니다.

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
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, classification_report
from typing import Dict, Tuple

# 경로 설정
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import load_config, get_training_config
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel

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


def analyze_probability_distribution(xgb_model: XGBoostDirectionModel, 
                                    X_val: np.ndarray, 
                                    y_val: np.ndarray) -> Dict:
    """XGBoost 예측 확률 분포 분석"""
    print("\n[최적화] 예측 확률 분포 분석 중...")
    
    # 예측 확률 계산
    proba = xgb_model.model.predict_proba(X_val)
    
    # 클래스별 확률 추출 (0=HOLD, 1=BUY, 2=SELL)
    hold_probs = proba[:, 0]
    buy_probs = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))
    sell_probs = proba[:, 2] if proba.shape[1] > 2 else np.zeros(len(proba))
    
    # 통계 정보
    stats = {
        'hold_mean': np.mean(hold_probs),
        'hold_std': np.std(hold_probs),
        'hold_min': np.min(hold_probs),
        'hold_max': np.max(hold_probs),
        'buy_mean': np.mean(buy_probs),
        'buy_std': np.std(buy_probs),
        'buy_min': np.min(buy_probs),
        'buy_max': np.max(buy_probs),
        'sell_mean': np.mean(sell_probs),
        'sell_std': np.std(sell_probs),
        'sell_min': np.min(sell_probs),
        'sell_max': np.max(sell_probs),
    }
    
    print(f"\n[최적화] 확률 분포 통계:")
    print(f"  HOLD: 평균={stats['hold_mean']:.3f}, 표준편차={stats['hold_std']:.3f}, 범위=[{stats['hold_min']:.3f}, {stats['hold_max']:.3f}]")
    print(f"  BUY:  평균={stats['buy_mean']:.3f}, 표준편차={stats['buy_std']:.3f}, 범위=[{stats['buy_min']:.3f}, {stats['buy_max']:.3f}]")
    print(f"  SELL: 평균={stats['sell_mean']:.3f}, 표준편차={stats['sell_std']:.3f}, 범위=[{stats['sell_min']:.3f}, {stats['sell_max']:.3f}]")
    
    return {
        'hold_probs': hold_probs,
        'buy_probs': buy_probs,
        'sell_probs': sell_probs,
        'stats': stats
    }


def find_optimal_thresholds(proba_dist: Dict, y_val: np.ndarray) -> Dict:
    """Precision-Recall 곡선을 사용하여 최적 임계값 찾기"""
    print("\n[최적화] 최적 임계값 계산 중...")
    
    buy_probs = proba_dist['buy_probs']
    sell_probs = proba_dist['sell_probs']
    
    # BUY 클래스에 대한 이진 분류 (BUY vs 나머지)
    y_buy = (y_val == 1).astype(int)
    
    # SELL 클래스에 대한 이진 분류 (SELL vs 나머지)
    y_sell = (y_val == 2).astype(int)
    
    # 임계값 범위 (0.3 ~ 0.9, 0.01 간격)
    thresholds = np.arange(0.3, 0.91, 0.01)
    
    results = []
    
    for threshold in thresholds:
        # BUY 예측
        buy_pred = (buy_probs > threshold) & (buy_probs > sell_probs) & (buy_probs > proba_dist['hold_probs'])
        
        # SELL 예측
        sell_pred = (sell_probs > threshold) & (sell_probs > buy_probs) & (sell_probs > proba_dist['hold_probs'])
        
        # 실제 레이블과 비교
        buy_precision = precision_score(y_buy, buy_pred, zero_division=0)
        buy_recall = recall_score(y_buy, buy_pred, zero_division=0)
        buy_f1 = f1_score(y_buy, buy_pred, zero_division=0)
        
        sell_precision = precision_score(y_sell, sell_pred, zero_division=0)
        sell_recall = recall_score(y_sell, sell_pred, zero_division=0)
        sell_f1 = f1_score(y_sell, sell_pred, zero_division=0)
        
        # 평균 F1-score (BUY와 SELL의 평균)
        avg_f1 = (buy_f1 + sell_f1) / 2.0
        
        # 거래 빈도 (거래 비율)
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
    
    # 거래 빈도가 너무 낮은 경우 (5% 미만) 고려
    # F1-score가 높으면서 거래 빈도가 적절한 임계값 찾기
    filtered_results = results_df[
        (results_df['trade_frequency'] >= 0.05) &  # 최소 5% 거래 빈도
        (results_df['avg_f1'] >= results_df['avg_f1'].max() * 0.9)  # 최대 F1의 90% 이상
    ]
    
    if len(filtered_results) > 0:
        # 거래 빈도와 F1-score의 균형 (가중 평균)
        filtered_results['balanced_score'] = (
            filtered_results['avg_f1'] * 0.7 + 
            filtered_results['trade_frequency'] * 0.3
        )
        balanced_idx = filtered_results['balanced_score'].idxmax()
        balanced_threshold = filtered_results.loc[balanced_idx, 'threshold']
    else:
        balanced_threshold = optimal_threshold
    
    print(f"\n[최적화] 최적 임계값 분석 결과:")
    print(f"  F1-score 최대화 임계값: {optimal_threshold:.3f}")
    print(f"    - BUY F1: {results_df.loc[optimal_idx, 'buy_f1']:.3f}")
    print(f"    - SELL F1: {results_df.loc[optimal_idx, 'sell_f1']:.3f}")
    print(f"    - 평균 F1: {results_df.loc[optimal_idx, 'avg_f1']:.3f}")
    print(f"    - 거래 빈도: {results_df.loc[optimal_idx, 'trade_frequency']:.2%}")
    
    if balanced_threshold != optimal_threshold:
        balanced_idx = filtered_results['balanced_score'].idxmax()
        print(f"\n  균형 잡힌 임계값: {balanced_threshold:.3f}")
        print(f"    - BUY F1: {filtered_results.loc[balanced_idx, 'buy_f1']:.3f}")
        print(f"    - SELL F1: {filtered_results.loc[balanced_idx, 'sell_f1']:.3f}")
        print(f"    - 평균 F1: {filtered_results.loc[balanced_idx, 'avg_f1']:.3f}")
        print(f"    - 거래 빈도: {filtered_results.loc[balanced_idx, 'trade_frequency']:.2%}")
    
    return {
        'optimal_threshold': optimal_threshold,
        'balanced_threshold': balanced_threshold,
        'results_df': results_df
    }


def visualize_threshold_analysis(proba_dist: Dict, optimization_results: Dict, save_dir: str):
    """임계값 분석 결과 시각화"""
    print("\n[최적화] 시각화 생성 중...")
    
    os.makedirs(save_dir, exist_ok=True)
    results_df = optimization_results['results_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 확률 분포 히스토그램
    axes[0, 0].hist(proba_dist['hold_probs'], bins=50, alpha=0.5, label='HOLD', edgecolor='black')
    axes[0, 0].hist(proba_dist['buy_probs'], bins=50, alpha=0.5, label='BUY', edgecolor='black')
    axes[0, 0].hist(proba_dist['sell_probs'], bins=50, alpha=0.5, label='SELL', edgecolor='black')
    axes[0, 0].axvline(optimization_results['optimal_threshold'], color='r', linestyle='--', 
                       label=f'최적 임계값: {optimization_results["optimal_threshold"]:.3f}')
    axes[0, 0].set_xlabel('예측 확률')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].set_title('예측 확률 분포')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1-score vs 임계값
    axes[0, 1].plot(results_df['threshold'], results_df['buy_f1'], 'o-', label='BUY F1', alpha=0.7, markersize=3)
    axes[0, 1].plot(results_df['threshold'], results_df['sell_f1'], 's-', label='SELL F1', alpha=0.7, markersize=3)
    axes[0, 1].plot(results_df['threshold'], results_df['avg_f1'], '^-', label='평균 F1', alpha=0.7, markersize=3, linewidth=2)
    axes[0, 1].axvline(optimization_results['optimal_threshold'], color='r', linestyle='--', 
                       label=f'최적: {optimization_results["optimal_threshold"]:.3f}')
    axes[0, 1].set_xlabel('임계값')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score vs 임계값')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall 곡선
    axes[1, 0].plot(results_df['buy_recall'], results_df['buy_precision'], 'o-', label='BUY', alpha=0.7, markersize=3)
    axes[1, 0].plot(results_df['sell_recall'], results_df['sell_precision'], 's-', label='SELL', alpha=0.7, markersize=3)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall 곡선')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 거래 빈도 vs F1-score
    scatter = axes[1, 1].scatter(results_df['trade_frequency'], results_df['avg_f1'], 
                                c=results_df['threshold'], cmap='viridis', 
                                s=50, alpha=0.6, edgecolors='black')
    axes[1, 1].axvline(0.05, color='r', linestyle='--', alpha=0.5, label='최소 거래 빈도 (5%)')
    axes[1, 1].scatter([results_df.loc[results_df['avg_f1'].idxmax(), 'trade_frequency']], 
                       [results_df.loc[results_df['avg_f1'].idxmax(), 'avg_f1']],
                       color='red', s=200, marker='*', label='최적점', edgecolors='black', linewidth=2)
    axes[1, 1].set_xlabel('거래 빈도')
    axes[1, 1].set_ylabel('평균 F1-Score')
    axes[1, 1].set_title('거래 빈도 vs F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='임계값')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_optimization.png'), dpi=150)
    print(f"[최적화] 시각화 저장: {os.path.join(save_dir, 'threshold_optimization.png')}")
    plt.close()


def recommend_thresholds(optimization_results: Dict, proba_dist: Dict) -> Dict:
    """최종 임계값 추천"""
    optimal = optimization_results['optimal_threshold']
    balanced = optimization_results['balanced_threshold']
    stats = proba_dist['stats']
    
    # 통계적 근거 기반 추천
    # BUY/SELL 평균 확률의 1.5 표준편차 이상
    buy_statistical = max(0.5, stats['buy_mean'] + 1.5 * stats['buy_std'])
    sell_statistical = max(0.5, stats['sell_mean'] + 1.5 * stats['sell_std'])
    
    # 최종 추천: F1 최적화와 통계적 근거의 평균
    recommended_buy = np.clip((optimal + balanced + buy_statistical) / 3.0, 0.4, 0.8)
    recommended_sell = np.clip((optimal + balanced + sell_statistical) / 3.0, 0.4, 0.8)
    
    # strong_signal_threshold는 추천값보다 0.1 높게
    recommended_strong = min(0.8, recommended_buy + 0.1)
    
    return {
        'buy_threshold': round(recommended_buy, 3),
        'sell_threshold': round(recommended_sell, 3),
        'strong_signal_threshold': round(recommended_strong, 3),
        'optimal_threshold': optimal,
        'balanced_threshold': balanced,
        'statistical_buy': buy_statistical,
        'statistical_sell': sell_statistical
    }


def main():
    """메인 함수"""
    print("=" * 60)
    print("[최적화] 임계값 최적화 시작")
    print("=" * 60)
    
    # 설정 로드
    config = load_config()
    
    # 결과 저장 디렉토리 (example_dir는 이미 상단에서 정의됨)
    save_dir = os.path.join(example_dir, 'results', 'threshold_optimization')
    os.makedirs(save_dir, exist_ok=True)
    
    # 학습된 모델 로드
    print("\n[최적화] 학습된 모델 로드 중...")
    models_dir = os.path.join(example_dir, 'models')
    xgb_model_path = os.path.join(models_dir, 'xgboost_direction_model.pkl')
    
    if not os.path.exists(xgb_model_path):
        print(f"[최적화] 오류: 학습된 모델을 찾을 수 없습니다: {xgb_model_path}")
        print("[최적화] 먼저 train_sample.py를 실행하여 모델을 학습하세요.")
        return
    
    xgb_config = config.get('models', {}).get('xgboost', {})
    xgb_model = XGBoostDirectionModel(xgb_config)
    xgb_model.load(xgb_model_path)
    print("[최적화] 모델 로드 완료")
    
    # 검증 데이터 준비
    print("\n[최적화] 검증 데이터 준비 중...")
    from ai_learning.train_sample import load_data, create_labels
    from feature_engineering.feature_engineering import FeatureEngineering
    
    data = load_data(config)
    if data is None:
        return
    
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    labels = create_labels(features_df, data_with_indicators['close'], config)
    
    # 검증 데이터 분할 (학습 데이터의 마지막 20%)
    training_config = get_training_config(config)
    train_test_split = training_config.get('train_test_split', 0.8)
    split_idx = int(len(features_df) * train_test_split)
    validation_split = training_config.get('validation_split', 0.8)
    val_split_idx = int(split_idx * validation_split)
    
    # 검증 데이터 (학습 데이터의 마지막 20%)
    X_val = features_df.iloc[val_split_idx:split_idx].values
    y_val = labels[val_split_idx:split_idx]
    
    print(f"[최적화] 검증 데이터: {len(X_val)}개 샘플")
    
    # 1. 확률 분포 분석
    proba_dist = analyze_probability_distribution(xgb_model, X_val, y_val)
    
    # 2. 최적 임계값 찾기
    optimization_results = find_optimal_thresholds(proba_dist, y_val)
    
    # 3. 시각화
    visualize_threshold_analysis(proba_dist, optimization_results, save_dir)
    
    # 4. 최종 추천
    recommendations = recommend_thresholds(optimization_results, proba_dist)
    
    print("\n" + "=" * 60)
    print("[최적화] 최종 추천 임계값")
    print("=" * 60)
    print(f"\n추천 설정 (sample_config.yaml에 적용):")
    print(f"  buy_threshold: {recommendations['buy_threshold']:.3f}")
    print(f"  sell_threshold: {recommendations['sell_threshold']:.3f}")
    print(f"  strong_signal_threshold: {recommendations['strong_signal_threshold']:.3f}")
    
    print(f"\n분석 근거:")
    print(f"  - F1-score 최적화 임계값: {recommendations['optimal_threshold']:.3f}")
    print(f"  - 균형 잡힌 임계값: {recommendations['balanced_threshold']:.3f}")
    print(f"  - 통계적 근거 (BUY): {recommendations['statistical_buy']:.3f}")
    print(f"  - 통계적 근거 (SELL): {recommendations['statistical_sell']:.3f}")
    
    # 결과를 파일로 저장
    result_file = os.path.join(save_dir, 'recommended_thresholds.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("임계값 최적화 결과\n")
        f.write("=" * 60 + "\n\n")
        f.write("추천 설정:\n")
        f.write(f"  buy_threshold: {recommendations['buy_threshold']:.3f}\n")
        f.write(f"  sell_threshold: {recommendations['sell_threshold']:.3f}\n")
        f.write(f"  strong_signal_threshold: {recommendations['strong_signal_threshold']:.3f}\n\n")
        f.write("분석 근거:\n")
        f.write(f"  - F1-score 최적화 임계값: {recommendations['optimal_threshold']:.3f}\n")
        f.write(f"  - 균형 잡힌 임계값: {recommendations['balanced_threshold']:.3f}\n")
        f.write(f"  - 통계적 근거 (BUY): {recommendations['statistical_buy']:.3f}\n")
        f.write(f"  - 통계적 근거 (SELL): {recommendations['statistical_sell']:.3f}\n")
    
    print(f"\n[최적화] 결과 저장: {result_file}")
    print("\n" + "=" * 60)
    print("[최적화] 임계값 최적화 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

