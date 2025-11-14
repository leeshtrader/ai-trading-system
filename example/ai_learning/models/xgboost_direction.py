"""
샘플: XGBoost 방향 예측 모델
XGBoost 모델 샘플 구현

주의: 본 모듈은 샘플/데모 목적입니다.

Sample: XGBoost Direction Prediction Model
Sample implementation of XGBoost model for direction prediction.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Optional
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
from matplotlib import rcParams, font_manager
import platform
from sklearn.metrics import confusion_matrix, classification_report

# 한글 폰트 설정
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

class XGBoostDirectionModel:
    """
    샘플: XGBoost 방향 예측 모델
    구조화된 피처를 입력받아 매수/매도/보유 방향을 예측합니다.
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 모델 설정 딕셔너리 (설정 파일에서 로드)
        """
        if config is None:
            # 기본값 (설정 파일을 찾을 수 없을 때만 사용)
            config = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.05,
                'min_child_weight': 1,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'num_class': 3,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
        
        self.config = config
        self.model = None
        print("[샘플] XGBoost 방향 예측 모델 초기화")
    
    def build_model(self):
        """모델 구조 생성"""
        # tree_method 설정 (GPU 사용 시 'gpu_hist', CPU 사용 시 'hist')
        tree_method = self.config.get('tree_method', 'hist')
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.05),
            min_child_weight=self.config.get('min_child_weight', 1),
            gamma=self.config.get('gamma', 0.1),
            reg_alpha=self.config.get('reg_alpha', 0.1),
            reg_lambda=self.config.get('reg_lambda', 1.0),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            objective='multi:softprob',
            num_class=self.config.get('num_class', 3),  # BUY, SELL, HOLD
            random_state=self.config.get('random_state', 42),
            eval_metric=self.config.get('eval_metric', 'mlogloss'),
            tree_method=tree_method
        )
        print(f"[샘플] XGBoost 모델 구조 생성 완료 (tree_method: {tree_method})")
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        모델 학습
        
        Args:
            X: 피처 데이터 (n_samples, n_features)
            y: 레이블 (0: HOLD, 1: BUY, 2: SELL)
            X_val: 검증 데이터 (선택사항)
            y_val: 검증 레이블 (선택사항)
        """
        if self.model is None:
            self.build_model()
        
        print(f"[샘플] XGBoost 학습 시작: {len(X)}개 샘플")
        
        # 클래스 불균형 처리: 클래스 가중치 계산 (개선)
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # 클래스별 가중치 계산 (SELL 클래스에 더 높은 가중치)
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # SELL 클래스(2)에 추가 가중치 부여 (성능이 낮으므로)
        weight_dict = dict(zip(classes, class_weights))
        if 2 in weight_dict:  # SELL 클래스
            weight_dict[2] *= 1.5  # SELL 가중치 1.5배 증가
        
        # 샘플 가중치 생성
        sample_weights = np.array([weight_dict[label] for label in y])
        
        # 검증 데이터가 있으면 사용
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
            val_sample_weights = np.array([weight_dict[label] for label in y_val])
        else:
            eval_set = [(X, y)]
            val_sample_weights = None
        
        # Early stopping 설정 (XGBoost 버전 호환성 문제로 인해 제거)
        # 참고: XGBoost 버전에 따라 early_stopping_rounds와 callbacks 지원이 다름
        # 안정성을 위해 early stopping은 비활성화
        # 필요시 n_estimators를 적절히 설정하여 과적합 방지
        
        # fit 실행
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )
        
        # 클래스 분포 출력
        unique, counts = np.unique(y, return_counts=True)
        print(f"[샘플] 학습 데이터 클래스 분포: {dict(zip(['HOLD', 'BUY', 'SELL'], counts))}")
        print("[샘플] XGBoost 학습 완료")
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        방향 예측
        
        Args:
            features: 피처 벡터 (1, n_features) 또는 (n_features,)
        
        Returns:
            방향 예측 결과 딕셔너리
        """
        if self.model is None:
            raise ValueError("[샘플] 모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        
        # 2D 배열로 변환
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 예측 확률
        proba = self.model.predict_proba(features)[0]
        
        # 클래스 순서: 0=HOLD, 1=BUY, 2=SELL (XGBoost의 기본 순서)
        # 실제로는 학습 시 레이블 인코딩에 따라 달라질 수 있음
        return {
            'buy_prob': float(proba[1] if len(proba) > 1 else proba[0]),
            'sell_prob': float(proba[2] if len(proba) > 2 else 0.0),
            'hold_prob': float(proba[0]),
            'trend_strength': float(abs(proba[1] - proba[2]) if len(proba) > 2 else 0.0),
            'confidence': float(max(proba))
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("[샘플] 저장할 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[샘플] 모델 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        self.model = joblib.load(filepath)
        print(f"[샘플] 모델 로드 완료: {filepath}")
    
    def visualize_results(self, X_test: np.ndarray, y_test: np.ndarray, 
                         feature_names: Optional[list] = None, 
                         save_dir: Optional[str] = None):
        """
        학습 결과 시각화
        
        Args:
            X_test: 테스트 피처 데이터
            y_test: 테스트 레이블
            feature_names: 피처 이름 리스트 (선택사항)
            save_dir: 이미지 저장 디렉토리 (선택사항, None이면 표시만)
        """
        if self.model is None:
            raise ValueError("[샘플] 모델이 학습되지 않았습니다.")
        
        print("\n[샘플] XGBoost 학습 결과 시각화 중...")
        
        # 예측 및 확률
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # 클래스 이름
        class_names = ['HOLD', 'BUY', 'SELL']
        
        # 1. 학습 곡선 (eval_result가 있는 경우)
        if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
            fig, ax = plt.subplots(figsize=(10, 6))
            for dataset_name, eval_result in self.model.evals_result_.items():
                for metric_name, metric_values in eval_result.items():
                    ax.plot(metric_values, label=f'{dataset_name}-{metric_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('XGBoost Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'xgboost_learning_curve.png'), dpi=150)
                print(f"[샘플] 학습 곡선 저장: {os.path.join(save_dir, 'xgboost_learning_curve.png')}")
            plt.close()
        
        # 2. 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='XGBoost Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # 숫자 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'xgboost_confusion_matrix.png'), dpi=150)
            print(f"[샘플] 혼동 행렬 저장: {os.path.join(save_dir, 'xgboost_confusion_matrix.png')}")
        plt.close()
        
        # 3. 예측 확률 분포
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, class_name in enumerate(class_names):
            axes[idx].hist(y_proba[:, idx], bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Probability')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{class_name} Probability Distribution')
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'xgboost_probability_distribution.png'), dpi=150)
            print(f"[샘플] 확률 분포 저장: {os.path.join(save_dir, 'xgboost_probability_distribution.png')}")
        plt.close()
        
        # 4. 특성 중요도
        feature_importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # 상위 15개만 표시
        top_n = min(15, len(feature_importance))
        indices = np.argsort(feature_importance)[-top_n:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), feature_importance[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('XGBoost Top Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'xgboost_feature_importance.png'), dpi=150)
            print(f"[샘플] 특성 중요도 저장: {os.path.join(save_dir, 'xgboost_feature_importance.png')}")
        plt.close()
        
        # 5. 분류 리포트 출력
        print("\n[샘플] XGBoost 분류 리포트:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # 6. 예측 확률 통계
        print("\n[샘플] 예측 확률 통계:")
        for idx, class_name in enumerate(class_names):
            print(f"  {class_name}: 평균={y_proba[:, idx].mean():.3f}, "
                  f"표준편차={y_proba[:, idx].std():.3f}, "
                  f"최대={y_proba[:, idx].max():.3f}, "
                  f"최소={y_proba[:, idx].min():.3f}")
        
        print("[샘플] XGBoost 시각화 완료")

