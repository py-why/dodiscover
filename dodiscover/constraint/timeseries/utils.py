import numpy as np


def convert_ts_df_to_multiindex(df, max_lag):
    n_samples = df.shape[0]

    # add lag column denoting which row corresponds to which lag
    # for a set time window of 'max_lag'
    q, r = divmod(n_samples, max_lag + 1)
    lag_list = [lag for lag in range(max_lag, -1, -1)]
    lag_col = q * lag_list + lag_list[:r]
    df["lag"] = lag_col

    # add naming for the time-series variables
    df.rename_axis("variable", axis=1, inplace=True)

    # compute which window each row in the dataframe is on
    df["windowed_sample"] = np.arange(len(df)) // (max_lag + 1)

    # convert lag to '-<lag>'
    df = df.assign(lag=-df["lag"])

    # create a multi-index with the first level as variable and
    # the second level as lag
    df = df.pivot(index="windowed_sample", columns="lag")
    return df
