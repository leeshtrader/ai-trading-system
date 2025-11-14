"""
샘플 프로젝트 실행 스크립트
AI Learning Module 학습 및 백테스트를 실행합니다.

주의: 본 스크립트는 샘플/데모 목적입니다.

Sample Project Execution Script
Executes AI Learning Module training and backtesting.

WARNING: This script is for sample/demo purposes only.
This is NOT investment advice. Use at your own risk.
"""
import sys
import os

# 경로 설정
example_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, example_dir)

from config.config_loader import load_config, get_data_config
from data.download_data import download_stock_data, save_data
from ai_learning.train_sample import main as train_main
import pandas as pd
from datetime import datetime

def main():
    """샘플 프로젝트 메인 실행"""
    print("=" * 60)
    print("[샘플] AI 트레이딩 시스템 샘플 프로젝트 실행")
    print("[샘플] AI Learning Module 학습 및 백테스트")
    print("=" * 60)
    
    # 설정 파일 로드
    config = load_config()
    data_config = get_data_config(config)
    
    # 1. 데이터 다운로드 (설정 파일에서 읽기)
    symbol = data_config.get('symbol', 'NVDA')
    start_date = data_config.get('start_date', '2019-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    output_dir = data_config.get('output_dir', 'data')
    output_file = data_config.get('output_file', f'{symbol.lower()}_data.csv')
    min_data_days = data_config.get('min_data_days', 730)
    
    data_path = os.path.join(example_dir, output_dir, output_file)
    
    # 오늘 날짜 확인
    today = datetime.now().date()
    
    # 데이터 파일이 있는지 확인하고, 최신 데이터인지 확인
    need_download = False
    if not os.path.exists(data_path):
        print(f"\n[샘플] 1단계: {symbol} 데이터 다운로드")
        need_download = True
    else:
        print(f"\n[샘플] 1단계: 기존 {symbol} 데이터 확인 중...")
        # 기존 데이터 로드하여 마지막 날짜 확인
        existing_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        last_date = existing_data.index[-1].date()
        data_days = (existing_data.index[-1] - existing_data.index[0]).days
        
        # 마지막 날짜가 오늘보다 오래되었거나, 데이터 기간이 부족하면 다운로드
        if last_date < today:
            print(f"[샘플] 기존 데이터의 마지막 날짜: {last_date}")
            print(f"[샘플] 오늘 날짜: {today}")
            print(f"[샘플] 최신 데이터를 다운로드합니다...")
            need_download = True
        elif data_days < min_data_days:
            print(f"[샘플] 기존 데이터가 {data_days}일로 부족합니다. 더 긴 기간의 데이터를 다운로드합니다.")
            need_download = True
        else:
            print(f"[샘플] 기존 데이터 사용 (마지막 날짜: {last_date})")
            data = existing_data
    
    if need_download:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data = download_stock_data(symbol, start_date, end_date)
        save_data(data, data_path)
        print(f"[샘플] 데이터 다운로드 완료: {len(data)}개 데이터 포인트")
    
    print(f"[샘플] 데이터 기간: {data.index[0].date()} ~ {data.index[-1].date()}")
    
    # 2. AI Learning Module 학습 및 백테스트
    print("\n[샘플] 2단계: AI Learning Module 학습 및 백테스트")
    train_results = train_main(config)
    
    # 결과 요약
    if train_results:
        backtest_results = train_results.get('backtest_results')
        if backtest_results:
            # 백테스트 결과 출력
            print("\n" + "=" * 60)
            print("[샘플] 백테스트 결과")
            print("=" * 60)
            print(f"초기 자본: {backtest_results.get('initial_balance', 0):,.0f}")
            print(f"최종 자본: {backtest_results.get('final_balance', 0):,.2f}")
            print(f"총 수익률: {backtest_results.get('total_return', 0):.2%}")
            print(f"총 거래 횟수: {backtest_results.get('total_trades', 0)}")
            print(f"승률: {backtest_results.get('win_rate', 0):.2%}")
            print(f"총 손익: {backtest_results.get('total_pnl', 0):,.2f}")
            print(f"평균 손익: {backtest_results.get('avg_pnl', 0):,.2f}")
            print(f"최대 수익: {backtest_results.get('max_profit', 0):,.2f}")
            print(f"최대 손실: {backtest_results.get('max_loss', 0):,.2f}")
            print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
            print(f"최대 낙폭: {backtest_results.get('max_drawdown', 0):,.0f} ({backtest_results.get('max_drawdown_pct', 0):.2%})")
            print("=" * 60)
            
            # 모델별 기여도 분석 제거: 최종 결정은 RL Agent가 내리므로 개별 모델 분석은 의미 없음
            # XGBoost와 LSTM은 RL Agent의 입력 정보로만 사용되며, 최종 포지션 크기 결정은 RL Agent가 수행
            
            # 거래 히스토리 요약
            trade_history = backtest_results.get('trade_history')
            if trade_history is not None and len(trade_history) > 0:
                print("\n[샘플] 거래 히스토리 (최근 10개):")
                print(trade_history[['timestamp', 'action', 'entry_price', 'exit_price', 'pnl', 'reason']].tail(10).to_string())
        else:
            print("\n[샘플] 백테스트 결과가 없습니다.")
    else:
        print("\n[샘플] 학습 결과가 없습니다.")
    
    print("\n" + "=" * 60)
    print("[샘플] 샘플 프로젝트 실행 완료")
    print("=" * 60)
    print("\n[샘플] 참고:")
    print("[샘플] - AI Learning Module은 AI 모델 학습 및 예측을 담당합니다.")
    print("[샘플] - 거래 실행 모듈은 실제 거래 실행을 담당합니다.")
    print("[샘플] - 성과 모니터링 모듈은 성과 분석 및 최적화를 담당합니다.")
    print("[샘플] - 본 샘플은 AI Learning Module까지만 구현되어 있습니다.")

if __name__ == "__main__":
    main()

