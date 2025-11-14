"""
설정 파일 로더
YAML 설정 파일을 로드하고 접근하는 유틸리티 모듈

Config File Loader
Utility module for loading and accessing YAML configuration files.
"""
import yaml
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

def _process_date_value(date_value: str, years_back: int = 5) -> str:
    """
    날짜 값을 처리합니다. "auto"인 경우 동적으로 계산합니다.
    
    Args:
        date_value: 날짜 값 ("auto" 또는 "YYYY-MM-DD" 형식)
        years_back: "auto"일 때 사용할 년수 (start_date용)
    
    Returns:
        처리된 날짜 문자열 ("YYYY-MM-DD" 형식)
    """
    if date_value.lower() == "auto":
        # end_date인 경우 오늘 날짜 반환
        if years_back == 0:
            return datetime.now().strftime("%Y-%m-%d")
        # start_date인 경우 years_back년 전 날짜 반환
        else:
            start_date = datetime.now() - timedelta(days=years_back * 365)
            return start_date.strftime("%Y-%m-%d")
    else:
        # 이미 날짜 형식인 경우 그대로 반환
        return date_value


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드하고 날짜 값을 처리합니다.
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        설정 딕셔너리 (날짜 값이 처리됨)
    """
    if config_path is None:
        # 기본 경로: example/config/sample_config.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'sample_config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 날짜 값 처리 (data 섹션)
    if 'data' in config:
        data_config = config['data']
        years_back = data_config.get('years_back', 5)
        
        # start_date 처리
        if 'start_date' in data_config:
            data_config['start_date'] = _process_date_value(
                data_config['start_date'], 
                years_back=years_back
            )
        
        # end_date 처리 (years_back=0으로 전달하여 오늘 날짜 반환)
        if 'end_date' in data_config:
            data_config['end_date'] = _process_date_value(
                data_config['end_date'], 
                years_back=0
            )
    
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    중첩된 설정 딕셔너리에서 점(.)으로 구분된 경로로 값을 가져옵니다.
    
    Args:
        config: 설정 딕셔너리
        key_path: 점으로 구분된 키 경로 (예: "models.xgboost.n_estimators")
        default: 기본값 (키를 찾을 수 없을 때)
    
    Returns:
        설정 값 또는 기본값
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """데이터 설정 반환"""
    return config.get('data', {})

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """특정 모델 설정 반환"""
    models = config.get('models', {})
    return models.get(model_name, {})

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """학습 설정 반환"""
    return config.get('training', {})

def get_labeling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """레이블 생성 설정 반환"""
    return config.get('labeling', {})

def get_backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """백테스트 설정 반환"""
    return config.get('backtest', {})

def get_ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """앙상블 설정 반환"""
    return config.get('ensemble', {})

def get_feature_engineering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """피처 엔지니어링 설정 반환"""
    return config.get('feature_engineering', {})

def get_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """시각화 설정 반환"""
    return config.get('visualization', {})

