"""
샘플 프로젝트용 데이터 다운로드 스크립트
야후 파이낸스에서 주식 데이터를 다운로드합니다.

주의: 본 스크립트는 샘플/데모 목적입니다.

Sample Data Download Script
Downloads stock data from Yahoo Finance.

WARNING: This script is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# 설정 파일 로드 시도
try:
    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, example_dir)
    from config.config_loader import load_config, get_data_config
    
    config = load_config()
    data_config = get_data_config(config)
    
    # 설정 파일에서 값 읽기
    SYMBOL = data_config.get('symbol', 'NVDA')
    START_DATE = data_config.get('start_date', '2019-01-01')
    END_DATE = data_config.get('end_date', '2024-12-31')
    OUTPUT_DIR = data_config.get('output_dir', 'data')
    OUTPUT_FILE = data_config.get('output_file', f'{SYMBOL.lower()}_data.csv')
except Exception:
    # 설정 파일을 찾을 수 없으면 기본값 사용
    SYMBOL = "NVDA"
    START_DATE = "2019-01-01"
    END_DATE = "2024-12-31"
    OUTPUT_DIR = "data"
    OUTPUT_FILE = "nvda_data.csv"

def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    야후 파이낸스에서 주식 데이터 다운로드
    
    Args:
        symbol: 주식 심볼 (예: "AAPL")
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        DataFrame: OHLCV 데이터
    """
    print(f"[샘플] {symbol} 데이터 다운로드 중... ({start_date} ~ {end_date})")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # 컬럼명 소문자로 변환
    data.columns = [col.lower() for col in data.columns]
    
    print(f"[샘플] 다운로드 완료: {len(data)}개 데이터 포인트")
    return data

def save_data(data: pd.DataFrame, output_path: str):
    """데이터를 CSV 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"[샘플] 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 데이터 다운로드
    data = download_stock_data(SYMBOL, START_DATE, END_DATE)
    
    # 데이터 저장
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    save_data(data, output_path)
    
    # 기본 정보 출력
    print("\n[샘플] 데이터 정보:")
    print(data.head())
    print(f"\n[샘플] 데이터 기간: {data.index[0]} ~ {data.index[-1]}")
    print(f"[샘플] 총 {len(data)}일의 데이터")

