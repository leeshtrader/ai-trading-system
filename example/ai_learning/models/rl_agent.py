"""
샘플: 강화학습 Agent
강화학습 모델 샘플 구현

주의: 본 모듈은 샘플/데모 목적입니다.

Sample: Reinforcement Learning Agent
Sample implementation of RL agent for trading decisions.

WARNING: This module is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import numpy as np
from typing import Dict, Optional
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용

# stable-baselines3 import 시도
STABLE_BASELINES3_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    # 더미 클래스 (에러 방지)
    class PPO:
        def __init__(self, *args, **kwargs):
            pass
        def learn(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return [0, 0.0], None
        def save(self, *args, **kwargs):
            pass
        @staticmethod
        def load(*args, **kwargs):
            return None

# gym/gymnasium import 시도
GYM_AVAILABLE = False
gym = None
spaces = None
try:
    # 먼저 gymnasium 시도 (최신 권장)
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        # gymnasium이 없으면 gym 시도
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        # 둘 다 없을 경우 더미 클래스
        class gym:
            class Env:
                pass
        
        class spaces:
            Box = None
        
        if STABLE_BASELINES3_AVAILABLE:
            print("[샘플] gym/gymnasium이 설치되지 않았습니다.")
            print("[샘플] gymnasium을 설치하세요: pip install gymnasium")
        else:
            print("[샘플] stable-baselines3가 설치되지 않았습니다.")
            print("[샘플] 강화학습 기능은 제한적으로 동작합니다.")

# Trading Environment 클래스
if GYM_AVAILABLE and STABLE_BASELINES3_AVAILABLE:
    class SimpleTradingEnv(gym.Env):
        """
        샘플: 간단한 트레이딩 환경
        강화학습을 위한 환경 정의
        """
        
        def __init__(self, price_data: np.ndarray, initial_balance: float = 100000, config: Dict = None):
            super(SimpleTradingEnv, self).__init__()
            
            if config is None:
                config = {}
            
            self.price_data = price_data
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.position = 0.0  # -1.0 ~ 1.0
            self.current_step = 0
            self.config = config
            
            # Action: Discrete action space (0=HOLD, 1=BUY, 2=SELL)
            self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
            self.position_size = config.get('position_size', 0.3)
            
            # Observation: [position, balance_ratio, price_normalized, price_change, ...]
            observation_space_size = config.get('observation_space_size', 12)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_space_size,),
                dtype=np.float32
            )
        
        def reset(self, seed=None, options=None):
            # gymnasium v0.28+ API 호환성
            self.balance = self.initial_balance
            self.position = 0.0
            self.current_step = 0
            observation = self._get_observation()
            info = {}
            return observation, info
        
        def step(self, action):
            # gymnasium v0.28+ API: (observation, reward, terminated, truncated, info)
            action_type = int(action)  # Discrete action: 0=HOLD, 1=BUY, 2=SELL
            position_size = self.position_size  # 고정 포지션 크기
            
            if self.current_step >= len(self.price_data) - 1:
                observation = self._get_observation()
                reward = 0.0
                terminated = True
                truncated = False
                info = {}
                return observation, reward, terminated, truncated, info
            
            current_price = self.price_data[self.current_step]
            next_price = self.price_data[self.current_step + 1]
            
            # 행동 실행
            if action_type == 1:  # BUY
                new_position = min(self.position + position_size, 1.0)
            elif action_type == 2:  # SELL
                new_position = max(self.position - position_size, -1.0)
            else:  # HOLD
                new_position = self.position
            
            # 수익률 계산
            price_change = (next_price - current_price) / current_price
            
            # 설정 파일에서 보상 파라미터 읽기
            reward_scale = self.config.get('reward_scale', 200)
            opportunity_cost_penalty = self.config.get('opportunity_cost_penalty', 50)
            avoidance_bonus = self.config.get('avoidance_bonus', 30)
            correct_direction_bonus = self.config.get('correct_direction_bonus', 100)
            wrong_direction_penalty = self.config.get('wrong_direction_penalty', 50)
            action_bonus = self.config.get('action_bonus', 0.5)
            transaction_cost = self.config.get('transaction_cost', 0.05)
            
            # 기본 보상: 포지션에 따른 수익률
            if self.position != 0.0:
                reward = self.position * price_change * reward_scale
            else:
                reward = 0.0
            
            # HOLD에 대한 기회비용 페널티 (가격이 오를 때 HOLD하면 손실)
            if action_type == 0:  # HOLD
                if price_change > 0.01:  # 가격이 1% 이상 오르면
                    reward -= abs(price_change) * opportunity_cost_penalty
                elif price_change < -0.01:  # 가격이 1% 이상 떨어지면
                    reward += abs(price_change) * avoidance_bonus
            
            # BUY/SELL 행동에 대한 보너스 (탐험 장려)
            if action_type == 1:  # BUY
                if price_change > 0:
                    reward += abs(price_change) * correct_direction_bonus
                else:
                    reward += price_change * wrong_direction_penalty
                reward += action_bonus
            elif action_type == 2:  # SELL
                if price_change < 0:
                    reward += abs(price_change) * correct_direction_bonus
                else:
                    reward += abs(price_change) * wrong_direction_penalty
                reward += action_bonus
            
            # 거래 비용
            if action_type != 0:
                reward -= transaction_cost
            
            self.position = new_position
            self.current_step += 1
            
            # 종료 조건
            terminated = (self.current_step >= len(self.price_data) - 1)
            truncated = False  # 샘플에서는 truncation 없음
            
            observation = self._get_observation()
            info = {}
            
            return observation, reward, terminated, truncated, info
        
        def _get_observation(self):
            """관측값 생성 (개선된 구성)"""
            if self.current_step >= len(self.price_data):
                return np.zeros(12)
            
            current_price = self.price_data[self.current_step]
            
            # 가격 변화율
            price_change = 0.0
            if self.current_step > 0:
                price_change = (current_price - self.price_data[self.current_step - 1]) / self.price_data[self.current_step - 1]
            
            # 단기/장기 이동평균 (간단한 추세 지표)
            ma_short = current_price
            ma_long = current_price
            if self.current_step >= 5:
                ma_short = np.mean(self.price_data[max(0, self.current_step-5):self.current_step+1])
            if self.current_step >= 20:
                ma_long = np.mean(self.price_data[max(0, self.current_step-20):self.current_step+1])
            
            # 변동성 (최근 10일)
            volatility = 0.01
            if self.current_step >= 10:
                recent_prices = self.price_data[max(0, self.current_step-10):self.current_step+1]
                if len(recent_prices) > 1:
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    volatility = np.std(returns) if len(returns) > 0 else 0.01
            
            # 모멘텀 (최근 5일 수익률)
            momentum = 0.0
            if self.current_step >= 5:
                momentum = (current_price - self.price_data[self.current_step - 5]) / self.price_data[self.current_step - 5]
            
            price_normalization = self.config.get('price_normalization', 200)
            observation_size = self.config.get('observation_space_size', 12)
            
            obs = [
                self.position,  # 현재 포지션
                self.balance / self.initial_balance,  # 정규화된 잔액
                current_price / price_normalization,  # 정규화된 가격
                price_change,  # 가격 변화율
                (ma_short - ma_long) / current_price if current_price > 0 else 0,  # 이동평균 차이
                momentum,  # 모멘텀
                volatility,  # 변동성
                min(self.current_step / 1000.0, 1.0),  # 시간 정규화
                0.5,  # direction_signal (실제로는 XGBoost 예측 사용)
                0.5,  # price_target (실제로는 LSTM 예측 사용)
                0.5,  # confidence
                1.0 if self.position != 0.0 else 0.0  # 포지션 보유 여부
            ]
            
            # observation_space_size에 맞게 조정
            if len(obs) < observation_size:
                obs.extend([0.0] * (observation_size - len(obs)))
            elif len(obs) > observation_size:
                obs = obs[:observation_size]
            
            return np.array(obs, dtype=np.float32)
else:
    # gym이 없을 경우 더미 클래스
    class SimpleTradingEnv:
        def __init__(self, *args, **kwargs):
            self.price_data = args[0] if args else np.array([100])
            self.balance = 100000
            self.position = 0.0
            self.current_step = 0
            self.action_space = None
            self.observation_space = None
        def reset(self):
            return np.zeros(12)
        def step(self, action):
            return np.zeros(12), 0.0, True, {}


class TradingRLAgent:
    """
    샘플: 강화학습 Agent
    XGBoost와 LSTM의 예측을 종합하여 최종 행동을 결정합니다.
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: 모델 설정 딕셔너리 (설정 파일에서 로드)
        """
        if config is None:
            config = {
                'learning_rate': 3e-4,
                'total_timesteps': 100000,
                'gamma': 0.99,
                'n_steps': 256,
                'batch_size': 128,
                'n_epochs': 10,
                'ent_coef': 0.05,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'clip_range': 0.2,
                'initial_balance': 100000,
                'position_size': 0.3,
                'observation_space_size': 12,
                'reward_scale': 200,
                'opportunity_cost_penalty': 50,
                'avoidance_bonus': 30,
                'correct_direction_bonus': 100,
                'wrong_direction_penalty': 50,
                'action_bonus': 0.5,
                'transaction_cost': 0.05,
                'price_normalization': 200
            }
        
        self.config = config
        self.agent = None
        self.training_rewards = []  # 학습 중 보상 기록
        self.training_episodes = []  # 에피소드 기록
        print("[샘플] 강화학습 Agent 초기화")
    
    def train(self, price_data: np.ndarray):
        """
        Agent 학습
        
        Args:
            price_data: 가격 데이터 배열
        """
        if not STABLE_BASELINES3_AVAILABLE or not GYM_AVAILABLE:
            print("[샘플] stable-baselines3 또는 gym이 설치되지 않아 강화학습을 건너뜁니다.")
            self.agent = None
            return
        
        try:
            print(f"[샘플] 강화학습 학습 시작: {len(price_data)}개 데이터 포인트")
            
            # 환경 생성 (설정 전달)
            initial_balance = self.config.get('initial_balance', 100000)
            env = SimpleTradingEnv(price_data, initial_balance=initial_balance, config=self.config)
            
            # PPO Agent 생성
            # learning_rate를 float로 변환 (YAML에서 문자열로 읽힐 수 있음)
            lr = self.config.get('learning_rate', 3e-4)
            if isinstance(lr, str):
                lr = float(lr)
            
            self.agent = PPO(
                'MlpPolicy',
                env,
                learning_rate=lr,
                n_steps=self.config.get('n_steps', 256),
                batch_size=self.config.get('batch_size', 128),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                ent_coef=self.config.get('ent_coef', 0.05),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                clip_range=self.config.get('clip_range', 0.2),
                verbose=1
            )
            
            # 학습 중 보상 수집을 위한 콜백 (간단한 방법)
            # 학습 (프로그레스바 활성화)
            self.agent.learn(total_timesteps=self.config['total_timesteps'], progress_bar=True)
            
            # 학습 후 평가를 통해 보상 곡선 생성
            self._collect_training_metrics(env)
            
            print("[샘플] 강화학습 학습 완료")
        except Exception as e:
            print(f"[샘플] 강화학습 학습 중 오류 발생: {e}")
            print("[샘플] 강화학습은 건너뛰고 기본 행동을 사용합니다.")
            self.agent = None
    
    def predict(self, state: Dict) -> Dict:
        """
        행동 예측
        
        Args:
            state: 상태 딕셔너리 (direction_signal, price_target, current_state 포함)
        
        Returns:
            행동 딕셔너리
        """
        if self.agent is None:
            # 학습되지 않은 경우 기본 행동 반환
            return {
                'action_type': 0,  # HOLD
                'position_size': 0.0,
                'confidence': 0.5
            }
        
        # 상태를 관측 벡터로 변환
        obs = self._state_to_observation(state)
        
        # 예측 (Discrete action이므로 단일 정수 반환)
        action, _ = self.agent.predict(obs, deterministic=True)
        
        position_size = self.config.get('position_size', 0.3)
        return {
            'action_type': int(action),  # Discrete action: 0, 1, 또는 2
            'position_size': position_size,
            'confidence': 0.7  # 샘플이므로 고정값
        }
    
    def _state_to_observation(self, state: Dict) -> np.ndarray:
        """상태를 관측 벡터로 변환 (12차원)"""
        direction = state.get('direction', {})
        price_target = state.get('price_target', {})
        current_state = state.get('current_state', {})
        
        current_price = current_state.get('current_price', 100.0)
        buy_prob = direction.get('buy_prob', 0.5)
        sell_prob = direction.get('sell_prob', 0.5)
        target_price = price_target.get('target_price', current_price * 1.02)
        stop_loss = price_target.get('stop_loss', current_price * 0.98)
        
        # 가격 변화율 (간단히 0으로 설정, 실제로는 이전 가격 필요)
        price_change = 0.0
        
        # 이동평균 차이 (간단히 0으로 설정, 실제로는 이전 가격들 필요)
        ma_diff = 0.0
        
        # 모멘텀 (간단히 0으로 설정)
        momentum = 0.0
        
        # 변동성
        volatility = current_state.get('market_volatility', 0.01)
        
        price_normalization = self.config.get('price_normalization', 200)
        observation_size = self.config.get('observation_space_size', 12)
        initial_balance = self.config.get('initial_balance', 100000)
        
        obs = [
            current_state.get('current_position', 0.0),  # 현재 포지션
            current_state.get('balance', initial_balance) / initial_balance,  # 정규화된 잔액
            current_price / price_normalization,  # 정규화된 가격
            price_change,  # 가격 변화율
            ma_diff,  # 이동평균 차이
            momentum,  # 모멘텀
            volatility,  # 변동성
            0.5,  # 시간 정규화 (샘플이므로 고정값)
            buy_prob,  # XGBoost BUY 확률
            sell_prob,  # XGBoost SELL 확률
            direction.get('confidence', 0.5),  # 신뢰도
            1.0 if current_state.get('current_position', 0.0) != 0.0 else 0.0  # 포지션 보유 여부
        ]
        
        # observation_space_size에 맞게 조정
        if len(obs) < observation_size:
            obs.extend([0.0] * (observation_size - len(obs)))
        elif len(obs) > observation_size:
            obs = obs[:observation_size]
        
        return np.array(obs, dtype=np.float32)
    
    def save(self, filepath: str):
        """모델 저장"""
        if self.agent is None:
            raise ValueError("[샘플] 저장할 모델이 없습니다.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)
        print(f"[샘플] 모델 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        try:
            # 환경이 필요하므로 임시 환경 생성
            initial_balance = self.config.get('initial_balance', 100000)
            temp_env = SimpleTradingEnv(np.array([100, 101, 102]), initial_balance=initial_balance, config=self.config)
            self.agent = PPO.load(filepath, env=temp_env)
            print(f"[샘플] 모델 로드 완료: {filepath}")
        except Exception as e:
            print(f"[샘플] 모델 로드 실패: {e}")
            self.agent = None
    
    def _collect_training_metrics(self, env):
        """학습 후 평가를 통해 메트릭 수집"""
        if self.agent is None:
            return
        
        # 여러 에피소드 평가
        episode_rewards = []
        episode_lengths = []
        episode_actions = []
        
        for _ in range(10):  # 10개 에피소드 평가
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            actions_in_episode = []
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                actions_in_episode.append(int(action))
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_actions.extend(actions_in_episode)
        
        self.training_rewards = episode_rewards
        self.training_episodes = list(range(len(episode_rewards)))
        self.training_actions = episode_actions
    
    def visualize_results(self, price_data: np.ndarray, save_dir: Optional[str] = None):
        """
        학습 결과 시각화
        
        Args:
            price_data: 가격 데이터 (평가용)
            save_dir: 이미지 저장 디렉토리 (선택사항)
        """
        if self.agent is None:
            print("[샘플] 학습된 RL Agent가 없어 시각화를 건너뜁니다.")
            return
        
        try:
            # 한글 폰트 설정
            import platform
            system = platform.system()
            if system == 'Windows':
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif system == 'Darwin':  # macOS
                plt.rcParams['font.family'] = 'AppleGothic'
            else:  # Linux
                try:
                    plt.rcParams['font.family'] = 'NanumGothic'
                except:
                    plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        print("\n[샘플] 강화학습 학습 결과 시각화 중...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. 에피소드별 보상 곡선
        if self.training_rewards:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.training_episodes, self.training_rewards, 'o-', label='에피소드 보상', color='blue', alpha=0.7)
            if len(self.training_rewards) > 1:
                # 이동평균
                window = min(5, len(self.training_rewards) // 2)
                if window > 1:
                    moving_avg = np.convolve(self.training_rewards, np.ones(window)/window, mode='valid')
                    ax.plot(self.training_episodes[window-1:], moving_avg, '--', label=f'이동평균 ({window})', color='red', linewidth=2)
            ax.set_xlabel('에피소드')
            ax.set_ylabel('누적 보상')
            ax.set_title('강화학습 에피소드별 보상 곡선')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_episode_rewards.png'), dpi=150)
                print(f"[샘플] 에피소드 보상 곡선 저장: {os.path.join(save_dir, 'rl_episode_rewards.png')}")
            plt.close()
        
        # 2. 평가 실행 (가격 데이터로)
        initial_balance = self.config.get('initial_balance', 100000)
        env = SimpleTradingEnv(price_data, initial_balance=initial_balance, config=self.config)
        obs, _ = env.reset()
        
        rewards = []
        actions = []
        positions = []
        prices = []
        done = False
        step_count = 0
        
        while not done and step_count < len(price_data) - 1:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(float(reward))
            actions.append(int(action))
            positions.append(float(env.position))
            prices.append(float(price_data[min(env.current_step, len(price_data)-1)]))
            
            step_count += 1
        
        # 3. 누적 보상 곡선
        if rewards:
            fig, ax = plt.subplots(figsize=(12, 6))
            cum_rewards = np.cumsum(rewards)
            ax.plot(cum_rewards, label='누적 보상', color='tab:blue', linewidth=2)
            if len(rewards) > 20:
                mov_avg = np.convolve(rewards, np.ones(20)/20, mode='valid')
                ax.plot(range(19, 19+len(mov_avg)), mov_avg, label='보상 이동평균 (20)', color='tab:orange', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('스텝')
            ax.set_ylabel('보상')
            ax.set_title('강화학습 평가: 누적 보상 및 이동평균')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_cumulative_rewards.png'), dpi=150)
                print(f"[샘플] 누적 보상 곡선 저장: {os.path.join(save_dir, 'rl_cumulative_rewards.png')}")
            plt.close()
        
        # 4. 액션 분포
        if actions:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 액션 타임라인
            axes[0].step(range(len(actions)), actions, where='post', color='tab:purple', linewidth=1)
            axes[0].set_yticks([0, 1, 2])
            axes[0].set_yticklabels(['HOLD', 'BUY', 'SELL'])
            axes[0].set_xlabel('스텝')
            axes[0].set_ylabel('액션')
            axes[0].set_title('액션 타임라인')
            axes[0].grid(True, alpha=0.3)
            
            # 액션 분포 (히스토그램)
            action_counts = [actions.count(0), actions.count(1), actions.count(2)]
            axes[1].bar(['HOLD', 'BUY', 'SELL'], action_counts, color=['gray', 'green', 'red'], alpha=0.7)
            axes[1].set_ylabel('빈도')
            axes[1].set_title('액션 분포')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_action_distribution.png'), dpi=150)
                print(f"[샘플] 액션 분포 저장: {os.path.join(save_dir, 'rl_action_distribution.png')}")
            plt.close()
        
        # 5. 가격과 포지션 변화
        if prices and positions:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # 가격 차트
            axes[0].plot(prices, color='tab:gray', label='가격', linewidth=1.5)
            # BUY/SELL 마커
            buy_indices = [i for i, a in enumerate(actions) if a == 1]
            sell_indices = [i for i, a in enumerate(actions) if a == 2]
            if buy_indices:
                axes[0].scatter(buy_indices, [prices[i] for i in buy_indices], 
                             color='tab:green', s=30, marker='^', label='BUY', zorder=5)
            if sell_indices:
                axes[0].scatter(sell_indices, [prices[i] for i in sell_indices], 
                             color='tab:red', s=30, marker='v', label='SELL', zorder=5)
            axes[0].set_ylabel('가격')
            axes[0].set_title('가격 및 거래 신호')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 포지션 변화
            axes[1].plot(positions, color='tab:brown', linewidth=1.5)
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[1].set_xlabel('스텝')
            axes[1].set_ylabel('포지션')
            axes[1].set_title('포지션 변화')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'rl_price_position.png'), dpi=150)
                print(f"[샘플] 가격 및 포지션 차트 저장: {os.path.join(save_dir, 'rl_price_position.png')}")
            plt.close()
        
        print("[샘플] 강화학습 시각화 완료")
