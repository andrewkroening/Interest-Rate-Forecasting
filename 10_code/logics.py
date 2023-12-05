#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def rate_modeler(
    ratedata,
    lookback,
    lookforward,
    hidden_layers,
    tol=1e-4,
    verbosity=True,
    save_data=False,
    save_model=False,
    visual=True,
):
    """Take the interest rate data and generate a model that takes in the lookback and lookforward ranges to generate predictions

    Parameters
    ----------
    ratedata : DataFrame
        The DataFrame containing the interest rate data
    lookback : int
        The number of days to look back
    lookforward : int
        The number of days to look forward
    hidden_layers : tuple
        The hidden layer sizes
    verbosity : bool
        The verbosity of the model
    save_data : bool
        Whether to save the data or not
    save_model : bool
        Whether to save the model or not
    visual : bool
        Whether to visualize the data or not

    Returns
    -------
    model_score : float
        The model score
    model_mse : float
        The model mean squared error
    """

    # set colors
    color0a = "#09EE90"
    color0b = "#008000"
    color1a = "#87CEFA"
    color1b = "#0000CD"
    color2a = "#F08080"
    color2b = "#B22222"

    print(f"-----Building Data-----")
    print(f"\n")

    # set the lookforward to -1 to account for indexing
    lookforward = lookforward - 1

    # set the lookback size
    lookback_size = lookback + lookforward

    # build the dataset based on the lookback and target
    predictor_rates = (
        ratedata.iloc[:lookback, 1:]
        .to_numpy()
        .reshape(-1, lookback, ratedata.shape[1] - 1)
    )

    # add the lookback data to the predictor_rates
    for i in range(1, len(ratedata) - lookback_size):
        predictor_rates = np.append(
            predictor_rates,
            ratedata.iloc[i : i + lookback, 1:]
            .to_numpy()
            .reshape(-1, lookback, ratedata.shape[1] - 1),
            axis=0,
        )

    # build the outcome dataset
    outcome_rates = ratedata.iloc[lookback_size:, 1:].to_numpy()

    # reshape the predictor_rates to match
    predictor_rates = predictor_rates.reshape(outcome_rates.shape[0], -1)

    # if save_data, combine the predictor_rates and outcome_rates into a single df and export to csv
    if save_data:
        # concat a single df with the first col from ratedata, predictor_rates, and outcome_rates
        rate_data = pd.concat(
            [
                ratedata.iloc[lookback_size:, 0],
                pd.DataFrame(predictor_rates),
                pd.DataFrame(outcome_rates),
            ],
            axis=1,
        )

        rate_data.to_csv(
            f"../20_intermediate_files/rate_data_{lookback}_{lookforward}.csv"
        )

        print(f"-----Data Saved-----")
        print(f"\n")

    # fit the model
    if save_model == True:
        print(f"-----Modeling Data-----")
        print(f"\n")

        # initiate the model
        rate_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            verbose=verbosity,
            max_iter=1000,
            random_state=42,
            tol=tol,
        )

        # fit the model
        rate_model.fit(predictor_rates, outcome_rates)

        # predictions
        predictions = rate_model.predict(predictor_rates)

        model_score = rate_model.score(predictor_rates, outcome_rates)
        model_mse = mean_squared_error(outcome_rates, predictions)

        # model scores
        print(f"-----Model Training Complete-----")

        # print scores and metrics
        print(f"Model Score: {model_score}")
        print(f"Model MSE: {model_mse}")

        # save the model
        joblib.dump(
            rate_model,
            f"../20_intermediate_files/rate_model_{lookback}_{lookforward}.pkl",
        )

    # if save_model == False, train with a train test split
    else:
        print(f"-----Modeling Data-----")
        print(f"\n")

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(
            predictor_rates, outcome_rates, random_state=42
        )

        # initiate the model
        rate_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            verbose=verbosity,
            max_iter=1000,
            random_state=42,
            tol=tol,
        )

        # fit the model
        rate_model.fit(X_train, y_train)

        # predictions
        predictions = rate_model.predict(X_test)

        # test the model
        model_score = rate_model.score(X_test, y_test)
        model_mse = mean_squared_error(y_test, predictions)

        # model scores
        print(f"-----Model Training Complete-----")

        # print scores and metrics
        print(f"Model Score: {model_score}")
        print(f"Model MSE: {model_mse}")
        print(f"\n")

    # if visual, visualize the data
    if visual:
        # get the full predictions
        predictions = rate_model.predict(predictor_rates)

        # compile the predictions and actual rates into a single df
        rate_viz_df = pd.DataFrame(
            {
                "Actual 3 Mo Rates": outcome_rates[:, 0],
                "Predicted 3 Mo Rates": predictions[:, 0],
                "Actual 3 Yr Rates": outcome_rates[:, 4],
                "Predicted 3 Yr Rates": predictions[:, 4],
                "Actual 10 Yr Rates": outcome_rates[:, 7],
                "Predicted 10 Yr Rates": predictions[:, 7],
            }
        )

        # plot the data
        plt.figure(figsize=(12, 8))

        # plot the actual rates
        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Actual 3 Mo Rates"],
            label="Actual 3 Mo Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color0a,
        )

        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Actual 3 Yr Rates"],
            label="Actual 3 Yr Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color1a,
        )

        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Actual 10 Yr Rates"],
            label="Actual 10 Yr Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color2a,
        )

        # plot the predicted rates
        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Predicted 3 Mo Rates"],
            label="Predicted 3 Mo Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color0b,
        )

        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Predicted 3 Yr Rates"],
            label="Predicted 3 Yr Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color1b,
        )

        plt.plot(
            ratedata.iloc[lookback_size:, 0],
            rate_viz_df["Predicted 10 Yr Rates"],
            label="Predicted 10 Yr Rates",
            alpha=0.7,
            linewidth=0.75,
            color=color2b,
        )

        # set a title
        plt.title("Actual vs. Predicted Rates")

        # set a y and x label
        plt.ylabel("Interest Rate")
        plt.xlabel("Year")

        # show a legend
        plt.legend(loc="upper right")

        # show the plot
        plt.show()

    return model_score, model_mse


def rate_forecast(lb, lf, df, save=True):
    """A function to forecast interest rates for 20 years into the future

    Parameters
    ----------
    lb : int
        The number of days used in model training for the lookback period
    lf : int
        The number of days used in model training for the forecast period
    df : dataframe
        The dataframe containing the historical interest rates

    Returns
    -------
    df_forecast
        A dataframe containing the forecasted interest rates
    """

    # define lb_size
    lb_size = lb + lf

    # load the model pkl
    model = joblib.load(f"../20_intermediate_files/rate_model_{lb}_{lf-1}.pkl")

    for i in range(5218):
        # find the predictor range by looking back from the end of the dataset lb_size rows and keeping the lb rows
        # this is the predictor range
        predictor_range = df.iloc[-lb_size:-lf].copy()

        # drop the first column
        predictor_range = predictor_range.drop(predictor_range.columns[0], axis=1)

        # # flatten into an array
        predictor_range = predictor_range.values.flatten()

        # predict the next day
        predicted_vals = model.predict([predictor_range])

        # turn sample vals into a df with cols [3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr]
        predicted_vals = pd.DataFrame(
            predicted_vals,
            columns=["3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr"],
        )

        # add sample vals to the df
        df = pd.concat([df, predicted_vals], axis=0)

    # reset the index
    df = df.reset_index(drop=True)

    # subset for the null dates and fill the Date with weekdays starting nov 1 2023
    df_forecast = df[df["Date"].isnull()].copy()
    df_forecast["Date"] = pd.date_range(
        start="11/1/2023", periods=len(df_forecast), freq="B"
    )

    # if save=True
    if save:
        # save df_forecast as csv
        df_forecast.to_csv(
            f"../20_intermediate_files/rate_forecast_{lb}_{lf-1}.csv", index=False
        )

    return df_forecast


def plot_forecast(direc, df, fcast):
    """Function to plot the total forecast

    Parameters:
        dir: str
            The directory to the forecast csv files
        df: dataframe
            The dataframe containing the historical interest rates
        fcast: list
            The forecast ranges

    Returns:
        None
    """

    # set colors
    color0a = "#09EE90"
    color0b = "#008000"
    color1a = "#87CEFA"
    color1b = "#0000CD"
    color2a = "#F08080"
    color2b = "#B22222"
    color3a = "#FFD700"
    color3b = "#FF8C00"

    # set the forecast ranges
    forecast_range = fcast
    center_lines = [color0b, color1b, color2b, color3b]
    fill_colors = [color0a, color1a, color2a, color3a]

    # set the font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="serif")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # create a figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # plot the original df
    ax.plot(df["Date"], df["3 Mo"], linewidth=1, color="black")
    ax.plot(df["Date"], df["6 Mo"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["1 Yr"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["2 Yr"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["3 Yr"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["5 Yr"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["7 Yr"], linewidth=1, color="grey")
    ax.plot(df["Date"], df["10 Yr"], linewidth=1, color="black")

    # find the files that end in csv
    files = [file for file in os.listdir(f"{direc}") if file.endswith(".csv")]

    # sort the files by the last number in the name
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # plot each forecast csv file
    for file in files:
        # read the csv file
        df_forecast = pd.read_csv(f"{direc}/{file}", parse_dates=["Date"])

        # get the data for a fill series
        x = df_forecast["Date"]
        y1 = df_forecast["3 Mo"]
        y2 = df_forecast["10 Yr"]

        # plot the fill series
        ax.fill_between(
            x, y1, y2, color=fill_colors[files.index(file)], alpha=0.2, linewidth=0
        )

        # plot the center line
        ax.plot(x, (y1 + y2) / 2, color=center_lines[files.index(file)], linewidth=1)

    # set a title
    ax.set_title(
        "20 Year Interest Rate Forecast",
        fontdict={"fontsize": 24, "fontweight": "bold"},
    )

    # set x and y labels
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Interest Rate (%)", fontsize=18)

    # make the tick labels larger
    ax.tick_params(axis="both", which="major", labelsize=15)

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # configure the legend
    handles = [
        plt.Line2D((0, 1), (0, 0), color=center_lines[files.index(file)], linewidth=1)
        for file in files
    ]

    # add the legend
    ax.legend(
        handles,
        forecast_range,
        loc="upper right",
        fontsize=16,
        title="Forecast Ranges",
        title_fontsize=18,
    )

    # set ylim
    ax.set_ylim(0, 12)

    # show the plot
    plt.show()


def plot_forecast_pct(direc, df, fcast):
    """Function to plot the total forecast

    Parameters:
        dir: str
            The directory to the forecast csv files
        df: dataframe
            The dataframe containing the historical interest rates
        fcast: list
            The forecast ranges

    Returns:
        None
    """

    # set colors
    color0a = "#09EE90"
    color0b = "#008000"
    color1a = "#87CEFA"
    color1b = "#0000CD"
    color2a = "#F08080"
    color2b = "#B22222"
    color3a = "#FFD700"
    color3b = "#FF8C00"

    # set the forecast ranges
    forecast_range = fcast
    center_lines = [color0b, color1b, color2b, color3b]
    fill_colors = [color0a, color1a, color2a, color3a]

    # set the font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="serif")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

    # create a figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # plot the original df
    ax.plot(df["Date"], df["3 Mo"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["6 Mo"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["1 Yr"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["2 Yr"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["3 Yr"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["5 Yr"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["7 Yr"], linewidth=1, color="grey", alpha=0.3)
    ax.plot(df["Date"], df["10 Yr"], linewidth=1, color="grey", alpha=0.3)

    # find the files in the 20_intermediate_forecast folder that end in csv
    files = [file for file in os.listdir(f"{direc}") if file.endswith(".csv")]

    # sort the files by the last number in the name
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # plot each forecast csv file
    for file in files:
        # read the csv file
        df_forecast = pd.read_csv(f"{direc}/{file}", parse_dates=["Date"])

        ax.plot(
            df_forecast["Date"],
            df_forecast["3 Mo"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["6 Mo"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["1 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["2 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["3 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["5 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["7 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )
        ax.plot(
            df_forecast["Date"],
            df_forecast["10 Yr"],
            linewidth=1,
            color=fill_colors[files.index(file)],
            alpha=0.3,
        )

    # set a title
    ax.set_title(
        "20 Year Interest Rate Forecast",
        fontdict={"fontsize": 24, "fontweight": "bold"},
    )

    # set x and y labels
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Interest Rate (%)", fontsize=18)

    # make the tick labels larger
    ax.tick_params(axis="both", which="major", labelsize=15)

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    # configure the legend
    handles = [
        plt.Line2D((0, 1), (0, 0), color=center_lines[files.index(file)], linewidth=1)
        for file in files
    ]

    # add the legend
    ax.legend(
        handles,
        forecast_range,
        loc="upper right",
        fontsize=16,
        title="Forecast Ranges",
        title_fontsize=18,
    )

    # set ylim
    ax.set_ylim(-0.6, 0.6)

    # show the plot
    plt.show()
