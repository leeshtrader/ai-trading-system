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
        
        # 비율 예측을 위한 출력 변환
        # 분할 매수/매도를 고려하여 넓은 범위 설정 (익절/손절 여유 확보)
        # 익절: 1.0~1.2 (최대 20% 상승), 손절: 0.8~1.0 (최대 20% 하락)
        self.ratio_scale_min = 0.8  # 넓은 범위 (손절 여유)
        self.ratio_scale_max = 1.2  # 넓은 범위 (익절 여유)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        
        import torch.nn.functional as F
        
        # target_ratio와 stop_loss_ratio를 비율 범위로 제한
        # 실제 데이터 분포에 맞게 조정:
        # - 실제 stop_loss_ratio 평균: 약 0.94, 범위: 0.85~0.985
        # - 실제 target_ratio 평균: 약 1.06, 범위: 1.03~1.10 (테스트 데이터 기준)
        # Tanh를 사용하여 -1~1 범위로 만든 후 실제 데이터 범위에 맞게 스케일링
        # 더 넓은 범위로 설정하여 변동성 학습 가능하도록
        ratio_0_raw = torch.tanh(output[:, 0]) * 0.1 + 1.06  # -1~1 → 0.96~1.16 (익절: 중앙값 1.06, 더 넓은 범위)
        ratio_1_raw = torch.tanh(output[:, 1]) * 0.08 + 0.94  # -1~1 → 0.86~1.02 (손절: 중앙값 0.94, 더 넓은 범위)
        
        # target_ratio > stop_loss_ratio 제약 적용
        # 먼저 기본 범위로 제한
        ratio_0 = torch.clamp(ratio_0_raw, 1.0, self.ratio_scale_max)  # 익절: 1.0~1.2
        ratio_1 = torch.clamp(ratio_1_raw, self.ratio_scale_min, 0.985)  # 손절: 0.8~0.985 (1.0 대신 0.985로 제한)
        
        # target_ratio와 stop_loss_ratio의 최소 차이 보장 (강화)
        min_diff = 0.03  # 최소 차이를 0.03으로 증가 (더 명확한 구분)
        # target_ratio가 stop_loss_ratio + min_diff보다 작으면 조정
        ratio_0 = torch.max(ratio_0, ratio_1 + min_diff)
        # stop_loss_ratio가 target_ratio - min_diff보다 크면 조정
        ratio_1 = torch.min(ratio_1, ratio_0 - min_diff)
        
        # 최종 제약 재확인 (차이가 유지되도록)
        ratio_0 = torch.max(ratio_0, ratio_1 + min_diff)
        ratio_1 = torch.min(ratio_1, ratio_0 - min_diff)
        
        # stop_loss_ratio가 0.985를 초과하지 않도록 강제 (중요!)
        ratio_1 = torch.clamp(ratio_1, self.ratio_scale_min, 0.985)
        
        # 나머지 출력값 처리
        output = torch.stack([
            ratio_0,  # target_ratio (>= 1.0)
            ratio_1,  # stop_loss_ratio (<= 1.0)
            F.relu(output[:, 2]),  # time_horizon (0 이상)
            F.sigmoid(output[:, 3]),  # confidence (0~1)
            F.relu(output[:, 4])  # volatility (0 이상)
        ], dim=1)
        
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
    
    def _initialize_weights(self):
        """가중치 초기화 개선"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Xavier 초기화 (더 나은 초기화)
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
    
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
        
        # 가중치 손실 함수 사용 (target_ratio와 stop_loss_ratio에 더 높은 가중치)
        criterion = nn.MSELoss(reduction='none')
        learning_rate = self.config.get('learning_rate', 0.001)
        optimizer_name = self.config.get('optimizer', 'Adam')
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
        
        # 학습률 스케줄러 추가
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"[샘플] LSTM 학습 시작: {len(sequences)}개 샘플, {epochs} 에폭")
        
        # 학습 곡선 저장용
        train_losses = []
        val_losses = []
        
        # 가중치 초기화 개선
        self._initialize_weights()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            
            # 가중치 손실: 각 출력 차원별로 개별 손실 계산 후 가중치 적용
            # criterion은 (batch_size, output_size) 형태 반환
            loss_per_dim = criterion(outputs, y)  # (batch_size, 5)
            
            # 각 차원별 가중치
            weight_per_dim = torch.tensor([2.0, 3.0, 0.5, 1.0, 0.5], device=y.device)  # [target, stop_loss, time, conf, vol]
            
            # 가중치 적용: (batch_size, 5) * (5,) -> (batch_size, 5)
            weighted_loss = loss_per_dim * weight_per_dim.unsqueeze(0)
            
            # 전체 손실: 모든 샘플과 차원에 대한 평균
            loss = weighted_loss.mean()
            
            # 변동성 페널티 추가 (예측값의 분산이 너무 작으면 페널티)
            if epoch > 5:  # 초기 학습 후에만 적용
                # target_ratio에 대한 variance penalty (강화)
                pred_std_target = torch.std(outputs[:, 0])  # target_ratio의 표준편차
                target_std_target = torch.std(y[:, 0])  # 실제 target_ratio의 표준편차
                if pred_std_target < target_std_target * 0.3:  # 예측 분산이 실제의 30% 미만이면 페널티
                    variance_penalty_target = (target_std_target - pred_std_target) * 2.0  # 1.0 -> 2.0으로 더 강화
                    loss = loss + variance_penalty_target
                
                # target_ratio가 1.0에 가까우면 강한 페널티 (익절 학습 개선)
                mean_target_pred = torch.mean(outputs[:, 0])
                mean_target_target = torch.mean(y[:, 0])
                if mean_target_pred < 1.02:  # 1.0에 너무 가까우면 (실제 평균은 1.06 근처)
                    target_low_penalty = (1.02 - mean_target_pred) * 10.0  # 매우 강한 페널티
                    loss = loss + target_low_penalty
                
                # target_ratio가 실제 범위(1.03~1.10) 밖에 있으면 페널티
                if mean_target_pred < 1.03 or mean_target_pred > 1.10:
                    target_range_penalty = abs(mean_target_pred - mean_target_target) * 5.0
                    loss = loss + target_range_penalty
                
                # stop_loss_ratio에 대한 variance penalty (강화)
                pred_std_stop = torch.std(outputs[:, 1])  # stop_loss_ratio의 표준편차
                target_std_stop = torch.std(y[:, 1])  # 실제 stop_loss_ratio의 표준편차
                if pred_std_stop < target_std_stop * 0.3:  # 예측 분산이 실제의 30% 미만이면 페널티 (10% -> 30%로 완화)
                    variance_penalty_stop = (target_std_stop - pred_std_stop) * 2.0  # 0.2 -> 2.0으로 강화
                    loss = loss + variance_penalty_stop
                
                # stop_loss_ratio가 0.985를 초과하면 매우 강한 페널티 (손절 학습 개선)
                mean_stop_pred = torch.mean(outputs[:, 1])
                mean_stop_target = torch.mean(y[:, 1])
                
                # 0.985 초과에 대한 강한 페널티
                if mean_stop_pred > 0.985:
                    excess_penalty = (mean_stop_pred - 0.985) * 20.0  # 매우 강한 페널티
                    loss = loss + excess_penalty
                
                # 실제 평균값과의 차이에 대한 페널티
                if abs(mean_stop_pred - mean_stop_target) > 0.01:  # 1% 이상 차이
                    mean_diff_penalty = abs(mean_stop_pred - mean_stop_target) * 10.0
                    loss = loss + mean_diff_penalty
                
                # stop_loss_ratio가 실제 범위(0.85~0.985) 밖에 있으면 페널티
                if mean_stop_pred < 0.85:
                    range_penalty = (0.85 - mean_stop_pred) * 5.0
                    loss = loss + range_penalty
            
            loss.backward()
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            # 검증 손실 계산
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    # 검증 손실도 가중치 적용 (동일한 방식)
                    val_loss_per_dim = criterion(val_outputs, y_val)
                    weight_per_dim = torch.tensor([2.0, 3.0, 0.5, 1.0, 0.5], device=y_val.device)
                    weighted_val_loss = val_loss_per_dim * weight_per_dim.unsqueeze(0)
                    val_loss = weighted_val_loss.mean().item()
                    val_losses.append(val_loss)
                self.model.train()
                
                # 학습률 스케줄러 업데이트
                scheduler.step(val_loss)
            else:
                val_losses.append(None)
            
            if (epoch + 1) % 5 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if val_losses[-1] is not None else ""
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[샘플] Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}{val_str}, LR: {current_lr:.6f}")
        
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

