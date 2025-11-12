"""
샘플: LSTM 가격 예측 모델
LSTM 모델 샘플 구현

주의: 본 모듈은 샘플/데모 목적입니다.

Sample: LSTM Price Prediction Model
Sample implementation of LSTM model for price prediction.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import pandas as pd

class LSTMModel(nn.Module):
    """샘플: LSTM 네트워크 구조"""
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, 
                 dropout: float = 0.2, fc_hidden_size: int = 16, output_size: int = 5):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class LSTMPriceModel:
    """
    샘플: LSTM 가격 예측 모델
    시계열 데이터를 입력받아 목표가, 손절가 등을 예측합니다.
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 모델 설정 딕셔너리 (설정 파일에서 로드)
        """
        if config is None:
            config = {
                'sequence_length': 20,
                'lstm_units': 32,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'device': 'cpu',
                'fc_hidden_size': 16,
                'output_size': 5,
                'default_target_ratio': 1.02,
                'default_stop_loss_ratio': 0.98,
                'unrealistic_range_min': 0.5,
                'unrealistic_range_max': 2.0
            }
        
        self.config = config
        self.model = None
        device_str = config.get('device', 'cpu')
        self.device = torch.device(device_str)
        print(f"[샘플] LSTM 가격 예측 모델 초기화 ({device_str} 사용)")
    
    def build_model(self, input_size: int):
        """모델 구조 생성"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.get('lstm_units', 32),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout_rate', 0.2),
            fc_hidden_size=self.config.get('fc_hidden_size', 16),
            output_size=self.config.get('output_size', 5)
        ).to(self.device)
        print(f"[샘플] LSTM 모델 구조 생성 완료 (input_size={input_size})")
    
    def train(self, sequences: np.ndarray, targets: np.ndarray, 
              sequences_val: Optional[np.ndarray] = None, 
              targets_val: Optional[np.ndarray] = None,
              epochs: int = 30):
        """
        모델 학습
        
        Args:
            sequences: 시계열 데이터 (n_samples, sequence_length, n_features)
            targets: 목표값 (n_samples, 5) - [target_price, stop_loss, time_horizon, confidence, volatility]
            sequences_val: 검증 시계열 데이터 (선택사항)
            targets_val: 검증 목표값 (선택사항)
            epochs: 학습 에폭 수
        """
        if self.model is None:
            raise ValueError("[샘플] 모델이 생성되지 않았습니다. build_model()을 먼저 호출하세요.")
        
        # 데이터를 텐서로 변환
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        # 검증 데이터
        X_val = None
        y_val = None
        if sequences_val is not None and targets_val is not None:
            X_val = torch.FloatTensor(sequences_val).to(self.device)
            y_val = torch.FloatTensor(targets_val).to(self.device)
        
        criterion = nn.MSELoss()
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer_name = self.config.get('optimizer', 'Adam')
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        print(f"[샘플] LSTM 학습 시작: {len(sequences)}개 샘플, {epochs} 에폭")
        
        # 학습 곡선 저장용
        train_losses = []
        val_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            # 검증 손실 계산
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    val_losses.append(val_loss)
                self.model.train()
            else:
                val_losses.append(None)
            
            if (epoch + 1) % 5 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if val_losses[-1] is not None else ""
                print(f"[샘플] Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}{val_str}")
        
        # 학습 곡선 저장
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        print("[샘플] LSTM 학습 완료")
    
    def predict(self, sequences: np.ndarray, current_price: float = None) -> Dict:
        """
        가격 예측
        
        Args:
            sequences: 시계열 데이터 (1, sequence_length, n_features) 또는 (sequence_length, n_features)
            current_price: 현재 가격 (예측값을 현재 가격 기준으로 변환하기 위해 필요)
        
        Returns:
            가격 예측 결과 딕셔너리
        """
        if self.model is None:
            raise ValueError("[샘플] 모델이 학습되지 않았습니다.")
        
        # 입력 형식 변환
        if sequences.ndim == 2:
            sequences = sequences.reshape(1, sequences.shape[0], sequences.shape[1])
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequences).to(self.device)
            outputs = self.model(X)
            outputs = outputs.cpu().numpy()[0]
        
        # 설정 파일에서 기본값 읽기
        default_target_ratio = self.config.get('default_target_ratio', 1.02)
        default_stop_loss_ratio = self.config.get('default_stop_loss_ratio', 0.98)
        unrealistic_range_min = self.config.get('unrealistic_range_min', 0.5)
        unrealistic_range_max = self.config.get('unrealistic_range_max', 2.0)
        
        # LSTM이 예측한 값은 절대 가격이므로, 현재 가격이 있으면 검증/조정
        # sequences의 마지막 시점의 close 가격을 추출 (정규화 전)
        # sequences는 [close, volume, rsi, macd, bb_position] 형태
        # close는 보통 첫 번째 컬럼이고, 정규화되어 있을 수 있음
        # 현재 가격이 제공되면 그것을 기준으로 사용
        if current_price is None:
            # sequences에서 현재 가격 추정 (정규화된 값일 수 있음)
            # 샘플에서는 close가 100으로 정규화되었다고 가정
            estimated_current = sequences[0, -1, 0] * 100 if sequences.shape[0] > 0 else 150.0
        else:
            estimated_current = current_price
        
        # 예측값이 현재 가격 범위 내에 있는지 확인하고 조정
        target_price_raw = float(outputs[0])
        stop_loss_raw = float(outputs[1])
        
        # LSTM이 학습한 데이터는 절대 가격이므로, 예측값도 절대 가격
        # 하지만 모델이 제대로 학습되지 않았을 수 있으므로, 현재 가격 기준으로 검증
        if target_price_raw < estimated_current * unrealistic_range_min or target_price_raw > estimated_current * unrealistic_range_max:
            # 예측값이 비현실적이면 현재 가격 기준으로 조정
            target_price = estimated_current * default_target_ratio
        else:
            target_price = target_price_raw
        
        if stop_loss_raw < estimated_current * unrealistic_range_min or stop_loss_raw > estimated_current * (2.0 - unrealistic_range_min):
            # 예측값이 비현실적이면 현재 가격 기준으로 조정
            stop_loss = estimated_current * default_stop_loss_ratio
        else:
            stop_loss = stop_loss_raw
        
        # 최종 검증: target_price는 stop_loss보다 커야 함
        if target_price <= stop_loss:
            target_price = estimated_current * default_target_ratio
            stop_loss = estimated_current * default_stop_loss_ratio
        
        return {
            'target_price': target_price,
            'stop_loss': stop_loss,
            'time_horizon': max(1, int(outputs[2])),  # 최소 1일
            'price_confidence': float(np.clip(outputs[3], 0, 1)),  # 0-1로 클리핑
            'volatility_forecast': float(max(0, outputs[4]))  # 음수 방지
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("[샘플] 저장할 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"[샘플] 모델 저장 완료: {filepath}")
    
    def load(self, filepath: str, input_size: int):
        """모델 로드"""
        self.build_model(input_size)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"[샘플] 모델 로드 완료: {filepath}")
    
    def visualize_results(self, sequences_test: np.ndarray, targets_test: np.ndarray,
                         save_dir: Optional[str] = None):
        """
        학습 결과 시각화
        
        Args:
            sequences_test: 테스트 시계열 데이터
            targets_test: 테스트 목표값
            save_dir: 이미지 저장 디렉토리 (선택사항)
        """
        if self.model is None:
            raise ValueError("[샘플] 모델이 학습되지 않았습니다.")
        
        print("\n[샘플] LSTM 학습 결과 시각화 중...")

        # 한글 폰트 설정 (시각화 글꼴 깨짐 방지)
        import platform
        from matplotlib import rcParams
        system = platform.system()
        if system == 'Windows':
            rcParams['font.family'] = 'Malgun Gothic'
        elif system == 'Darwin':
            rcParams['font.family'] = 'AppleGothic'
        else:
            rcParams['font.family'] = 'DejaVu Sans'
        rcParams['axes.unicode_minus'] = False
        
        # 입력 데이터 검증 및 변환
        sequences_test = np.array(sequences_test)
        targets_test = np.array(targets_test)
        
        # targets_test가 1D인 경우 2D로 변환
        if targets_test.ndim == 1:
            # 5개 값이 연속으로 있는 경우 reshape
            if len(targets_test) % 5 == 0:
                targets_test = targets_test.reshape(-1, 5)
            else:
                raise ValueError(f"[샘플] targets_test의 형태가 올바르지 않습니다: {targets_test.shape}")
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(sequences_test).to(self.device)
            predictions = self.model(X_test).cpu().numpy()
        
        # 크기 확인
        if predictions.shape[0] != targets_test.shape[0]:
            print(f"[샘플] 경고: 예측 크기({predictions.shape[0]})와 실제값 크기({targets_test.shape[0]})가 다릅니다.")
            min_size = min(predictions.shape[0], targets_test.shape[0])
            predictions = predictions[:min_size]
            targets_test = targets_test[:min_size]
        
        # 1. 학습 곡선
        if hasattr(self, 'train_losses') and self.train_losses:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_losses, label='Train Loss', color='blue')
            if self.val_losses and self.val_losses[0] is not None:
                ax.plot(self.val_losses, label='Validation Loss', color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('LSTM Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'lstm_learning_curve.png'), dpi=150)
                print(f"[샘플] 학습 곡선 저장: {os.path.join(save_dir, 'lstm_learning_curve.png')}")
            plt.close()
        
        # 2. 예측값 vs 실제값 비교 (target_price / stop_loss 시각화 차별화)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # target_price
        axes[0, 0].scatter(targets_test[:, 0], predictions[:, 0], alpha=0.5, s=10)
        axes[0, 0].plot([targets_test[:, 0].min(), targets_test[:, 0].max()],
                       [targets_test[:, 0].min(), targets_test[:, 0].max()], 
                       'r--', lw=2, label='Perfect Prediction')
        # 이상치로 인한 축 왜곡 방지 (5~95 분위 범위로 제한)
        tp_min = np.percentile(np.concatenate([targets_test[:, 0], predictions[:, 0]]), 5)
        tp_max = np.percentile(np.concatenate([targets_test[:, 0], predictions[:, 0]]), 95)
        axes[0, 0].set_xlim(tp_min, tp_max)
        axes[0, 0].set_ylim(tp_min, tp_max)
        # 지표 주석 (MAE/R2)
        from sklearn.metrics import r2_score, mean_absolute_error
        tp_mae = mean_absolute_error(targets_test[:, 0], predictions[:, 0])
        tp_r2 = r2_score(targets_test[:, 0], predictions[:, 0])
        axes[0, 0].text(0.02, 0.95, f"MAE: {tp_mae:.3f}\nR²: {tp_r2:.3f}", transform=axes[0, 0].transAxes,
                        fontsize=9, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        axes[0, 0].set_xlabel('Actual Target Price')
        axes[0, 0].set_ylabel('Predicted Target Price')
        axes[0, 0].set_title('Target Price: Predicted vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # stop_loss (hexbin으로 차별화)
        hb = axes[0, 1].hexbin(targets_test[:, 1], predictions[:, 1], gridsize=35, cmap='Blues', mincnt=1)
        axes[0, 1].plot([targets_test[:, 1].min(), targets_test[:, 1].max()],
                        [targets_test[:, 1].min(), targets_test[:, 1].max()], 
                        'r--', lw=2, label='Perfect Prediction')
        sl_min = np.percentile(np.concatenate([targets_test[:, 1], predictions[:, 1]]), 5)
        sl_max = np.percentile(np.concatenate([targets_test[:, 1], predictions[:, 1]]), 95)
        axes[0, 1].set_xlim(sl_min, sl_max)
        axes[0, 1].set_ylim(sl_min, sl_max)
        sl_mae = mean_absolute_error(targets_test[:, 1], predictions[:, 1])
        sl_r2 = r2_score(targets_test[:, 1], predictions[:, 1])
        axes[0, 1].text(0.02, 0.95, f"MAE: {sl_mae:.3f}\nR²: {sl_r2:.3f}", transform=axes[0, 1].transAxes,
                        fontsize=9, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        cb = fig.colorbar(hb, ax=axes[0, 1])
        cb.set_label('Count')
        axes[0, 1].set_xlabel('Actual Stop Loss')
        axes[0, 1].set_ylabel('Predicted Stop Loss')
        axes[0, 1].set_title('Stop Loss: Density (Hexbin) vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 예측 오차 분포 (target_price)
        errors_target = predictions[:, 0] - targets_test[:, 0]
        axes[1, 0].hist(errors_target, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Error (Target Price)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Target Price Prediction Error Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', label='Zero Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 예측 오차 분포 (stop_loss)
        errors_stop = predictions[:, 1] - targets_test[:, 1]
        axes[1, 1].hist(errors_stop, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error (Stop Loss)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Stop Loss Prediction Error Distribution')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Zero Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'lstm_prediction_comparison.png'), dpi=150)
            print(f"[샘플] 예측 비교 저장: {os.path.join(save_dir, 'lstm_prediction_comparison.png')}")
        plt.close()
        
        # 3. 시계열 예측 결과 (샘플 50개만)
        sample_size = min(50, len(targets_test))
        sample_indices = np.linspace(0, len(targets_test)-1, sample_size, dtype=int)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # target_price 시계열
        axes[0].plot(sample_indices, targets_test[sample_indices, 0], 
                    'o-', label='Actual Target Price', alpha=0.7, markersize=4)
        axes[0].plot(sample_indices, predictions[sample_indices, 0], 
                    's-', label='Predicted Target Price', alpha=0.7, markersize=4)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Price')
        axes[0].set_title('Target Price: Time Series Prediction (Sample)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # stop_loss 시계열
        axes[1].plot(sample_indices, targets_test[sample_indices, 1], 
                    'o-', label='Actual Stop Loss', alpha=0.7, markersize=4)
        axes[1].plot(sample_indices, predictions[sample_indices, 1], 
                    's-', label='Predicted Stop Loss', alpha=0.7, markersize=4)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Price')
        axes[1].set_title('Stop Loss: Time Series Prediction (Sample)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'lstm_time_series_prediction.png'), dpi=150)
            print(f"[샘플] 시계열 예측 저장: {os.path.join(save_dir, 'lstm_time_series_prediction.png')}")
        plt.close()
        
        # 4. 통계 출력
        print("\n[샘플] LSTM 예측 통계:")
        print(f"  Target Price:")
        print(f"    MAE: {np.mean(np.abs(errors_target)):.4f}")
        print(f"    RMSE: {np.sqrt(np.mean(errors_target**2)):.4f}")
        print(f"    평균 오차: {np.mean(errors_target):.4f}")
        print(f"  Stop Loss:")
        print(f"    MAE: {np.mean(np.abs(errors_stop)):.4f}")
        print(f"    RMSE: {np.sqrt(np.mean(errors_stop**2)):.4f}")
        print(f"    평균 오차: {np.mean(errors_stop):.4f}")
        
        print("[샘플] LSTM 시각화 완료")

