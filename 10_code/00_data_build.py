#!/usr/bin/env python3

# Imports

import pandas as pd
import os

# Set working directory
data_dir = "../00_data/_annual_data/"

# Read the csvs in a loop and build one df from them
df = pd.DataFrame()

# loop the files
for file in os.listdir(data_dir):
    # only get the csvs
    if file.endswith(".csv"):
        # make a temp_df and append
        df_temp = pd.read_csv(data_dir + file)

        df = pd.concat([df, df_temp], axis=0)

# covert the date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# sort by date
df = df.sort_values(by="Date")

# Save the df
df.to_csv("../00_data/base_data.csv", index=False)
