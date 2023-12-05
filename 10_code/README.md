## This directory contains the code used for the project

**Contents:**

- [`00_data_build.py`](00_data_build.py): A python script that combined the original `.csv` files of interest rate data into one single `.csv` file.

- [`00_data_explore.ipynb`](00_data_explore.ipynb): A jupyter notebook that explores the data and creates some visualizations.

- [`00_sp_data_explore.ipynb`](00_sp_data_explore.ipynb): A jupyter notebook that explores the S&P 500 data and creates some visualizations.

- [`00_sp_interest_merge_logged.ipynb`](00_sp_interest_merge_logged.ipynb): A jupyter notebook that cleans the S&P 500 data and creates a `.csv` file with logged data.

- [`00_sp_interest_merge_unlogged.ipynb`](00_sp_interest_merge_unlogged.ipynb): A jupyter notebook that cleans the S&P 500 data and creates a `.csv` file with absolute values.

- [`01_rate_pcts_diffs.ipynb`](01_rate_pcts_diffs.ipynb): A jupyter notebook that calculates the percentage change and difference between the interest rates.

- [`logics_sp500.py`](logics_sp500.py): A python script that contains the logic for the S&P 500 models.

- [`logics.py`](logics.py): A python script that contains the logic for the models.

- [`ML_model_diff.ipynb`](ML_model_diff.ipynb): A jupyter notebook that contains the code for training the MLP Regressor and building a forecast for the interest rates using difference.

- [`ML_model_pct.ipynb`](ML_model_pct.ipynb): A jupyter notebook that contains the code for training the MLP Regressor and building a forecast for the interest rates using percentage change.

- [`ML_model_rate.ipynb`](ML_model_rate.ipynb): A jupyter notebook that contains the code for training the MLP Regressor and building a forecast for the interest rates using absolute value.

- [`ML_model_sp500_rate.ipynb`](ML_model_sp500_rate.ipynb): A jupyter notebook that contains the code for training the MLP Regressor and building a forecast for the S&P 500.

- [`ML_sp500_visualize.ipynb`](ML_sp500_visualize.ipynb): A jupyter notebook that contains the code for visualizing the MLP Regressor forecasts for the S&P 500.

- [`ML_visualize.ipynb`](ML_visualize.ipynb): A jupyter notebook that contains the code for visualizing the MLP Regressor forecasts.
