# TradingExperiments

This repository contains the code for the experiments conducted to assess the profitability
of the RL agents and environments defined in the StrategyBuilder repository.

Each script contains one experiment. There are two main directories:

- **multi_symbol_experiment**: Contains experiments that test profitability on multiple symbols.
- **sentiment_experiment**: Contains experiments that test profitability of the sentiment score
  of having sentiment score as a feature on a single symbol.

To be able to run these experiment the following requirements must be met:

- Install requirements by running `pip install -r requirements.txt`
- Start MLFlow by running `mlflow ui`

After these requirements are met, you can run the experiments by running the scripts.
