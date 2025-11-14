"""
보상 함수 파라미터 최적화 (Sharpe Ratio 기반)
Optuna를 사용하여 보상 함수 파라미터를 Sharpe Ratio 최적화로 결정

주의: 본 스크립트는 샘플/데모 목적입니다.
"""
import sys
import os
import numpy as np
import pandas as pd
import optuna
from typing import Dict
import yaml
import copy

# 경로 설정
example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, example_dir)

from config.config_loader import load_config
from data.download_data import download_stock_data, save_data
from feature_engineering.feature_engineering import FeatureEngineering
from ai_learning.models.xgboost_direction import XGBoostDirectionModel
from ai_learning.models.lstm_price import LSTMPriceModel
from ai_learning.models.rl_agent import TradingRLAgent
from ai_learning.ensemble.role_based_ensemble import RoleBasedEnsemble
from ai_learning.backtest import Backtester
# train_sample 함수들 import
sys.path.insert(0, os.path.join(example_dir, 'ai_learning'))
from train_sample import load_data, create_labels, create_price_targets


def objective(trial: optuna.Trial, config: Dict, data_with_indicators: pd.DataFrame,
              features_df: pd.DataFrame, sequences: np.ndarray, labels: np.ndarray,
              price_targets: np.ndarray, train_test_split: float) -> float:
    """
    Optuna 목적 함수: Sharpe Ratio 최대화
    
    Args:
        trial: Optuna trial 객체
        config: 기본 설정 딕셔너리
        data_with_indicators: 기술적 지표가 포함된 데이터
        features_df: XGBoost용 피처
        sequences: LSTM용 시퀀스
        labels: 레이블
        price_targets: 가격 목표
        train_test_split: 학습/테스트 분할 비율
    
    Returns:
        Sharpe Ratio (최대화 목표)
    """
    trial_num = trial.number + 1
    print(f"\n[최적화] Trial {trial_num} 시작...")
    
    # 설정 복사
    trial_config = copy.deepcopy(config)
    
    # 보상 함수 파라미터를 Optuna로 최적화
    rl_config = trial_config.get('models', {}).get('rl', {})
    
    # reward_scale: 기본 스케일 (100~300 범위)
    reward_scale = trial.suggest_float('reward_scale', 100.0, 300.0, log=True)
    rl_config['reward_scale'] = reward_scale
    
    # 상대적 비율로 다른 파라미터 결정 (이론적 근거: reward_scale 기준 상대 비율)
    # position_risk_penalty_scale: reward_scale의 0.2~0.6배
    risk_penalty_ratio = trial.suggest_float('risk_penalty_ratio', 0.2, 0.6)
    rl_config['position_risk_penalty_scale'] = reward_scale * risk_penalty_ratio
    
    # opportunity_cost_penalty: reward_scale의 0.1~0.4배
    opp_cost_ratio = trial.suggest_float('opp_cost_ratio', 0.1, 0.4)
    rl_config['opportunity_cost_penalty'] = reward_scale * opp_cost_ratio
    
    # avoidance_bonus: reward_scale의 0.1~0.4배
    avoidance_ratio = trial.suggest_float('avoidance_ratio', 0.1, 0.4)
    rl_config['avoidance_bonus'] = reward_scale * avoidance_ratio
    
    # correct_direction_bonus: reward_scale의 0.3~0.7배
    correct_dir_ratio = trial.suggest_float('correct_dir_ratio', 0.3, 0.7)
    rl_config['correct_direction_bonus'] = reward_scale * correct_dir_ratio
    
    # wrong_direction_penalty: reward_scale의 0.1~0.4배
    wrong_dir_ratio = trial.suggest_float('wrong_dir_ratio', 0.1, 0.4)
    rl_config['wrong_direction_penalty'] = reward_scale * wrong_dir_ratio
    
    # profit_taking_bonus_scale: reward_scale의 0.5~1.5배
    profit_taking_ratio = trial.suggest_float('profit_taking_ratio', 0.5, 1.5)
    rl_config['profit_taking_bonus_scale'] = reward_scale * profit_taking_ratio
    
    # loss_expansion_penalty_scale: reward_scale의 0.3~0.8배
    loss_expansion_ratio = trial.suggest_float('loss_expansion_ratio', 0.3, 0.8)
    rl_config['loss_expansion_penalty_scale'] = reward_scale * loss_expansion_ratio
    
    # LSTM 기반 보상 파라미터
    lstm_bonus_base = trial.suggest_float('lstm_bonus_base', 2.0, 10.0)
    lstm_bonus_scale = trial.suggest_float('lstm_bonus_scale', 5.0, 15.0)
    rl_config['lstm_position_bonus'] = lstm_bonus_base
    rl_config['lstm_position_bonus_scale'] = lstm_bonus_scale
    rl_config['lstm_position_penalty_scale'] = lstm_bonus_base * 0.5  # 보너스의 절반
    
    # XGBoost 신호 일치 보너스
    signal_match_ratio = trial.suggest_float('signal_match_ratio', 0.02, 0.1)
    rl_config['signal_match_bonus'] = reward_scale * signal_match_ratio
    
    # 포지션 다양성 보너스
    diversity_ratio = trial.suggest_float('diversity_ratio', 0.005, 0.03)
    rl_config['position_diversity_bonus'] = reward_scale * diversity_ratio
    
    # 학습/테스트 분할 (train_sample.py와 동일한 방식)
    split_idx = int(len(features_df) * train_test_split)
    
    # sequence_length 가져오기 (train_sample.py와 동일)
    lstm_config = trial_config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    
    X_train = features_df.iloc[:split_idx].values
    X_test = features_df.iloc[split_idx:].values
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    
    # 시퀀스는 sequence_length개 앞서 시작 (train_sample.py와 동일)
    seq_train = sequences[:split_idx-sequence_length]
    seq_test = sequences[split_idx-sequence_length:]
    target_train = price_targets[:split_idx-sequence_length]
    target_test = price_targets[split_idx-sequence_length:]
    
    try:
        # XGBoost 학습 (빠른 학습을 위해 에포크 수 줄임)
        print(f"[최적화] Trial {trial_num}: XGBoost 학습 중...")
        xgb_config = trial_config.get('models', {}).get('xgboost', {})
        xgb_model = XGBoostDirectionModel(xgb_config)
        
        val_split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:val_split_idx]
        X_val_split = X_train[val_split_idx:]
        y_train_split = y_train[:val_split_idx]
        y_val_split = y_train[val_split_idx:]
        
        xgb_model.train(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # LSTM 학습 (빠른 학습을 위해 에포크 수 줄임)
        print(f"[최적화] Trial {trial_num}: LSTM 학습 중...")
        lstm_config = trial_config.get('models', {}).get('lstm', {})
        lstm_config['epochs'] = 20  # 최적화를 위해 줄임
        lstm_model = LSTMPriceModel(lstm_config)
        input_size = lstm_config.get('input_size', 5)
        lstm_model.build_model(input_size=input_size)
        
        val_split_idx = int(len(seq_train) * 0.8)
        seq_train_split = seq_train[:val_split_idx]
        seq_val_split = seq_train[val_split_idx:]
        target_train_split = target_train[:val_split_idx]
        target_val_split = target_train[val_split_idx:]
        
        lstm_model.train(seq_train_split, target_train_split,
                        sequences_val=seq_val_split,
                        targets_val=target_val_split,
                        epochs=lstm_config['epochs'])
        
        # RL Agent 학습 (빠른 학습을 위해 timesteps 줄임)
        print(f"[최적화] Trial {trial_num}: RL Agent 학습 중...")
        price_array = data_with_indicators['close'].values
        rl_agent = TradingRLAgent(rl_config)
        
        train_price_array = price_array[:split_idx]
        train_features = features_df.iloc[:split_idx]
        # 시퀀스는 sequence_length개 앞서 시작 (train_sample.py와 동일)
        train_sequences = sequences[:split_idx-sequence_length] if len(sequences) > split_idx-sequence_length else sequences
        
        # 최적화를 위해 timesteps 줄임
        original_timesteps = rl_config.get('total_timesteps', 100000)
        rl_config['total_timesteps'] = min(20000, original_timesteps)  # 최적화용으로 줄임
        
        rl_agent.train(
            price_data=train_price_array,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            features_data=train_features,
            sequences_data=train_sequences
        )
        
        # 백테스트 실행
        print(f"[최적화] Trial {trial_num}: 백테스트 실행 중...")
        backtest_config = trial_config.get('backtest', {})
        initial_balance = backtest_config.get('initial_balance', 100000)
        backtester = Backtester(initial_balance=initial_balance, config=trial_config)
        
        feature_eng = FeatureEngineering(trial_config)
        backtest_results = backtester.run_backtest(
            data=data_with_indicators,
            xgb_model=xgb_model,
            lstm_model=lstm_model,
            rl_agent=rl_agent,
            feature_eng=feature_eng,
            start_idx=split_idx,
            end_idx=len(data_with_indicators)
        )
        
        # Sharpe Ratio 반환 (최대화 목표)
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
        
        print(f"[최적화] Trial {trial_num} 완료: Sharpe Ratio = {sharpe_ratio:.4f}")
        
        # 음수 Sharpe Ratio는 페널티
        if sharpe_ratio <= 0:
            return -100.0  # 큰 페널티
        
        return sharpe_ratio
        
    except Exception as e:
        print(f"[최적화] Trial {trial_num} 실패: {e}")
        import traceback
        traceback.print_exc()
        return -100.0  # 실패 시 큰 페널티


def optimize_reward_parameters(config_path: str = None, n_trials: int = 20):
    """
    보상 함수 파라미터 최적화 (Sharpe Ratio 기반)
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로)
        n_trials: Optuna trial 수
    """
    print("=" * 60)
    print("[최적화] 보상 함수 파라미터 최적화 시작 (Sharpe Ratio 기반)")
    print("=" * 60)
    
    # 설정 로드
    if config_path is None:
        config_path = os.path.join(example_dir, 'config', 'sample_config.yaml')
    
    config = load_config(config_path)
    
    # 데이터 로드
    print("\n[최적화] 데이터 로드 중...")
    data = load_data(config)
    if data is None:
        print("[최적화] 데이터 로드 실패")
        return None
    
    # 피처 엔지니어링
    print("[최적화] 피처 엔지니어링 중...")
    feature_eng = FeatureEngineering(config)
    data_with_indicators = feature_eng.calculate_technical_indicators(data)
    data_with_indicators = data_with_indicators.dropna()
    
    features_df = feature_eng.prepare_features_for_xgboost(data_with_indicators)
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = lstm_config.get('sequence_length', 20)
    sequences = feature_eng.prepare_sequences_for_lstm(data_with_indicators, sequence_length=sequence_length)
    
    # 레이블 및 타겟 생성
    training_config = config.get('training', {})
    train_test_split = training_config.get('train_test_split', 0.8)
    labels = create_labels(features_df, data_with_indicators['close'], config)
    price_targets = create_price_targets(data_with_indicators['close'], data_with_indicators, config)
    
    print(f"[최적화] 데이터 준비 완료: {len(data_with_indicators)}개 데이터 포인트")
    print(f"[최적화] Optuna 최적화 시작: {n_trials}개 trial")
    print("[최적화] 주의: 각 trial마다 전체 모델 학습이 필요하므로 시간이 오래 걸릴 수 있습니다.")
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='maximize',  # Sharpe Ratio 최대화
        study_name='reward_function_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
    )
    
    # 진행 상황 출력을 위한 callback
    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"[최적화] Trial {trial.number + 1}/{n_trials} 완료: Sharpe Ratio = {trial.value:.4f}")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"[최적화] Trial {trial.number + 1}/{n_trials} 실패")
    
    # 최적화 실행
    print(f"\n[최적화] 총 {n_trials}개 trial 시작...")
    study.optimize(
        lambda trial: objective(
            trial, config, data_with_indicators,
            features_df, sequences, labels, price_targets, train_test_split
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[callback]
    )
    
    # 최적 파라미터 출력
    print("\n" + "=" * 60)
    print("[최적화] 최적화 완료")
    print("=" * 60)
    print(f"\n최고 Sharpe Ratio: {study.best_value:.4f}")
    print("\n최적 파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # 최적 파라미터를 설정 파일 형식으로 변환
    best_params = study.best_params
    reward_scale = best_params['reward_scale']
    
    optimal_rl_config = {
        'reward_scale': reward_scale,
        'position_risk_penalty_scale': reward_scale * best_params['risk_penalty_ratio'],
        'opportunity_cost_penalty': reward_scale * best_params['opp_cost_ratio'],
        'avoidance_bonus': reward_scale * best_params['avoidance_ratio'],
        'correct_direction_bonus': reward_scale * best_params['correct_dir_ratio'],
        'wrong_direction_penalty': reward_scale * best_params['wrong_dir_ratio'],
        'profit_taking_bonus_scale': reward_scale * best_params['profit_taking_ratio'],
        'loss_expansion_penalty_scale': reward_scale * best_params['loss_expansion_ratio'],
        'lstm_position_bonus': best_params['lstm_bonus_base'],
        'lstm_position_bonus_scale': best_params['lstm_bonus_scale'],
        'lstm_position_penalty_scale': best_params['lstm_bonus_base'] * 0.5,
        'signal_match_bonus': reward_scale * best_params['signal_match_ratio'],
        'position_diversity_bonus': reward_scale * best_params['diversity_ratio']
    }
    
    # 결과 저장
    results_dir = os.path.join(example_dir, 'results', 'optimization')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'optimal_reward_params.yaml')
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump({
            'best_sharpe_ratio': float(study.best_value),
            'optimal_parameters': optimal_rl_config,
            'raw_optuna_params': {k: float(v) for k, v in study.best_params.items()}
        }, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n[최적화] 결과 저장: {results_file}")
    
    return study.best_params, optimal_rl_config, config_path


def update_config_reward_params(optimal_rl_config: Dict, config_path: str = None):
    """
    최적화된 보상 함수 파라미터를 config 파일에 자동으로 업데이트
    
    Args:
        optimal_rl_config: 최적화된 RL 설정 딕셔너리
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
    """
    if config_path is None:
        config_path = os.path.join(example_dir, 'config', 'sample_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"[최적화] 경고: 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    
    # YAML 파일 읽기
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # rl 섹션 찾기
    rl_start_idx = None
    rl_end_idx = None
    in_rl = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # rl 섹션 시작 찾기
        if stripped.startswith('rl:') and not in_rl:
            rl_start_idx = i
            in_rl = True
            indent_level = len(line) - len(line.lstrip())
            continue
        
        if in_rl:
            # 다음 최상위 섹션 시작 시 종료
            current_indent = len(line) - len(line.lstrip())
            if stripped and not stripped.startswith('#') and current_indent <= indent_level and i > rl_start_idx:
                rl_end_idx = i
                break
    
    # rl 섹션이 끝나지 않았으면 파일 끝까지
    if in_rl and rl_end_idx is None:
        rl_end_idx = len(lines)
    
    if rl_start_idx is None:
        print(f"[최적화] 경고: rl 섹션을 찾을 수 없습니다.")
        return
    
    # 파라미터 업데이트할 항목들
    param_mappings = {
        'reward_scale': 'reward_scale',
        'position_risk_penalty_scale': 'position_risk_penalty_scale',
        'opportunity_cost_penalty': 'opportunity_cost_penalty',
        'avoidance_bonus': 'avoidance_bonus',
        'correct_direction_bonus': 'correct_direction_bonus',
        'wrong_direction_penalty': 'wrong_direction_penalty',
        'profit_taking_bonus_scale': 'profit_taking_bonus_scale',
        'loss_expansion_penalty_scale': 'loss_expansion_penalty_scale',
        'lstm_position_bonus': 'lstm_position_bonus',
        'lstm_position_bonus_scale': 'lstm_position_bonus_scale',
        'lstm_position_penalty_scale': 'lstm_position_penalty_scale',
        'signal_match_bonus': 'signal_match_bonus',
        'position_diversity_bonus': 'position_diversity_bonus'
    }
    
    # 업데이트된 라인 생성
    updated_lines = []
    param_updated = {key: False for key in param_mappings.keys()}
    
    for i, line in enumerate(lines):
        if i < rl_start_idx or (rl_end_idx is not None and i >= rl_end_idx):
            # rl 섹션 밖이면 그대로 추가
            updated_lines.append(line)
        else:
            # rl 섹션 내
            stripped = line.strip()
            updated = False
            
            for param_key, config_key in param_mappings.items():
                if param_key in optimal_rl_config:
                    # 파라미터가 포함된 라인 찾기 (정확한 매칭)
                    # 예: "reward_scale: 200" 또는 "  reward_scale: 200"
                    if (stripped.startswith(config_key + ':') or 
                        (config_key + ':') in stripped and not stripped.startswith('#')):
                        # 기존 값 업데이트
                        indent = len(line) - len(line.lstrip())
                        value = optimal_rl_config[param_key]
                        # 주석이 있으면 유지, 없으면 추가
                        comment = "  # 자동 최적화됨 (Sharpe Ratio 기반)"
                        if '#' in line:
                            # 기존 주석 유지하되 최적화 표시 추가
                            comment_idx = line.find('#')
                            existing_comment = line[comment_idx:].rstrip()
                            if '자동 최적화' not in existing_comment:
                                comment = f"  {existing_comment} [자동 최적화됨]"
                            else:
                                comment = f"  {existing_comment}"
                        updated_lines.append(f"{' ' * indent}{config_key}: {value:.4f}{comment}\n")
                        param_updated[param_key] = True
                        updated = True
                        break
            
            if not updated:
                updated_lines.append(line)
    
    # 누락된 파라미터 추가 (rl 섹션 끝에 추가)
    if not all(param_updated.values()):
        last_rl_line = rl_end_idx - 1 if rl_end_idx else len(updated_lines) - 1
        while last_rl_line >= 0 and (last_rl_line >= len(updated_lines) or 
                                      (updated_lines[last_rl_line].strip() == '' or 
                                       updated_lines[last_rl_line].strip().startswith('#'))):
            last_rl_line -= 1
        
        if last_rl_line >= 0:
            indent = len(updated_lines[last_rl_line]) - len(updated_lines[last_rl_line].lstrip())
            for param_key, config_key in param_mappings.items():
                if param_key in optimal_rl_config and not param_updated[param_key]:
                    value = optimal_rl_config[param_key]
                    updated_lines.insert(last_rl_line + 1, 
                                       f"{' ' * indent}{config_key}: {value:.4f}  # 자동 최적화됨 (Sharpe Ratio 기반)\n")
    
    # 파일 쓰기
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"\n[최적화] 설정 파일 자동 업데이트 완료: {config_path}")
    print("[최적화] 최적화된 파라미터가 config 파일에 반영되었습니다.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='보상 함수 파라미터 최적화 (Sharpe Ratio 기반)')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로')
    parser.add_argument('--n_trials', type=int, default=20, help='Optuna trial 수')
    
    args = parser.parse_args()
    
    optimize_reward_parameters(config_path=args.config, n_trials=args.n_trials)

