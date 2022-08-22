# Third party libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sns_displot(series, title, x, y):
    """
    Display a sns_displot with mode, median and mean.

    Parameters
    ----------
    series: pandas.Series
        Dataset series to be analyzed.

    title: str
        Figure title.

    x: str
        Chart x label

    y: str
        Chart y label

    Returns
    -------
    Plot Seaborn displot chart
    """

    # plotting data on distribution chart
    plt.figure(figsize=(10, 5))
    chart = sns.displot(
        # data=X_train,
        x=series,
        kind="kde",
        fill=True,
        height=5,
        aspect=1.5,
    )

    ##calculate statistics
    median = series.median()
    mean = round(series.mean(), 2)
    mode = series.mode()[0]

    # add vertical line to show statistical info
    plt.axvline(x=mode, color="b", ls=":", lw=2.5, label=f"Mode:{mode:,}")
    plt.axvline(
        x=median, color="black", ls="--", lw=2, label=f"Median: {median:,}"
    )
    plt.axvline(x=mean, color="r", ls="-", lw=1.5, label=f"Mean: {mean:,}")

    # customize plot
    chart.set(title=title)
    plt.xlabel(x, fontsize=10, labelpad=25)
    plt.ylabel(y, fontsize=10, labelpad=25)
    plt.legend(fontsize=12, shadow=True)

    return plt.show()


def missing_values(df):
    """
    Create dataframes with NaN and empty values. Both with features indicating
    the sum on NaN or Empty values and corresponding percentage of total.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to analize NaN and empty values.

    Returns
    -------
    Print both dataframes.
    """

    # build a dataframe with nan values by headers
    nan = pd.DataFrame(df.isna().sum(), columns=["NaN Values"])

    # calculate percentage
    nan["% of NaN"] = round(nan["NaN Values"] / (df.shape[0]), 4) * 100

    # sort values descending
    nan = nan.sort_values(by="% of NaN", ascending=False)

    # build a dataframe with empty values by headers
    empty = pd.DataFrame(df.eq(" ").sum(), columns=["Empty Values"])

    # calculate percentage
    empty["% of empty"] = round(empty["Empty Values"] / (df.shape[0]), 4) * 100

    # sort values descending
    empty = empty.sort_values(by="% of empty", ascending=False)

    # print features with nan values
    return print(
        f'{nan[nan["% of NaN"]>0]}\n\n' f'{empty[empty["% of empty"]>0]}'
    )


def series_count(series):
    """
    Create a dataframe with two features from series to visualize
    quantity of classes, sum of each class value and it percentage of total.

    Parameters
    ----------
    series : pandas.Series
        Dataset series to be analyzed.

    Returns
    -------
    Dataframe
    """

    series_count = series.value_counts(dropna=False)
    series_perc = round(
        series.value_counts(normalize=True, dropna=False) * 100, 2
    )

    return pd.DataFrame({"Count": series_count, "%": series_perc})
