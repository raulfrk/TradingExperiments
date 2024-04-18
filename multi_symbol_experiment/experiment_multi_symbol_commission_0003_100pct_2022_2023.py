"""
Experiment: 0.03 % commission, testing SNP500 between 2022 and 2023
"""
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlflow
from gymnasium import register

from strategybuilder.backtesting.multi_day_backtesting import multi_day_backtesting
from strategybuilder.defaults.data.candle_data_indicators import generate_candle_data_pipeline, \
    CANDLE_DATA_OBSERVABLE_COLUMNS, CANDLE_DATA_OBSERVABLE_WITH_SENTIMENT_COLUMNS
from strategybuilder.defaults.data.news_sentiment import generate_news_with_sentiment_pipeline
from strategybuilder.rl.agent import create_taining_envs, train_agent, BEST_HYPERPARAMS, get_input_weights
from strategybuilder.rl.env.long_only_env import LongOnlyStockTradingEnv

import warnings

# Disable all warnings
warnings.filterwarnings("ignore")
def run_experiment(train, test, symbol, required_headline_content, commission, initial_cash, obs_columns, max_size=None):
    mlflow.log_param("symbol", symbol)
    mlflow.log_param("required_headline_content", required_headline_content)
    mlflow.log_param("commission", commission)
    mlflow.log_param("initial_cash", initial_cash)
    mlflow.log_param("obs_columns", obs_columns)
    mlflow.log_param("max_size", max_size)
    mlflow.log_params(BEST_HYPERPARAMS)


    register(
        id="LongOnlyStockTradingEnv-v0",
        entry_point=LongOnlyStockTradingEnv,
    )
    train_env, test_env = create_taining_envs(train, test, "LongOnlyStockTradingEnv-v0", obs_columns, 15, commission,
                                                use_subproc=False, reset_at_random=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        agent = train_agent(train_env, test_env, save_path=temp_dir, eval_freq=4000, n_eval_episodes=50,
                            max_timesteps=len(train), checkpoint_save_freq=100_000 // 15, with_reporting=True,
                            **BEST_HYPERPARAMS)

        # Compute input weights importance of best model
        importance = get_input_weights(temp_dir/"best_model.zip", test_env)
        # Cast values to float
        importance = {k: float(v) for k, v in importance.items()}
        mlflow.log_dict(importance, "input_weights.json")

        all_stats, all_trades = multi_day_backtesting(test, initial_cash, obs_columns, "LongOnlyStockTradingEnv-v0", temp_dir/"best_model.zip", verbose=True, max_size=max_size, commission=commission)
        all_stats.sort_index(inplace=True)

        selected_cols_all_stats = [col for col in all_stats.columns if not col.startswith("_")]

        step = 0
        for _, trade in all_stats[selected_cols_all_stats].iterrows():
            dct = trade.to_dict()
            dct = {k.replace("[$]", "dollars").
                   replace("[%]", "pct").
                   replace(" ", "_").
                   replace("(Ann.)","").
                   replace(".","").
                   replace("&","").
                   replace("#","n"): float(v) for k, v in dct.items() if type(v) in [int, float]}
            mlflow.log_metrics(dct, step=step)
            step += 1
        mlflow.log_metric("final_balance", all_stats["Equity Final [$]"].iloc[-1])
        all_stats[selected_cols_all_stats].to_feather(temp_dir/"all_stats.feather")
        all_trades.to_feather(temp_dir/"all_trades.feather")
        mlflow.log_artifacts(temp_dir)



if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("multi_symbol_snp500_max_size_100pct_0003_commission_2022_2023")
    runs = 1
    symbols = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "BRK.B", "LLY", "AVGO", "JPM"]
    with_sentiment_decay = False
    for symbol in symbols:
        bar_data_pipeline = generate_candle_data_pipeline(symbol, datetime(2010, 1, 1), datetime(2024, 2, 15),
                                                          with_sentiment=False)

        data = bar_data_pipeline.transform(None)

        train = data.loc[datetime(2010, 9, 13):datetime(2022, 1, 15)]
        test = data.loc[datetime(2022, 1, 15):datetime(2023, 12, 30)]

        for i in range(runs):

            with mlflow.start_run():

                run_experiment(train, test, symbol, [], 0.0003, 100_000, CANDLE_DATA_OBSERVABLE_COLUMNS)
