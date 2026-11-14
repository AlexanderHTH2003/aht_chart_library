import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def aht_line_chart(
    data: pd.DataFrame | pd.Series,
    y_column: str | list |  None = None,
    chart_title: str | None = None,
    chart_subtitle: str | None = None,
    start_point: int | float | str | None = None,
    end_point: int | float | str | None = None,
    logscale: bool = False,
    show_mean: bool = False,
    show_bands: dict | bool = False, 
    show_moving_average: int | bool = False, 
    show_rangeselector: bool = True,
    plotly_template: str | None = "plotly_dark",
    chart_width: int | None = None,
    chart_height: int | None = None,

):
    """
    Generates an interactive line chart using Plotly Express.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The input data containing the time series to plot.
    y_column : str, list, or None, default=None
        The column name(s) to plot on the y-axis. If None, all numeric
        columns (excluding the index) will be plotted.
    chart_title : str or None, default=None
        The title of the chart. If None, a default title based on
        `y_column` is used.
    chart_subtitle : str or None, default=None
        The subtitle of the chart.
    start_point : int, float, str, or None, default=None
        The starting index/date for the data to be plotted. If None,
        it defaults to the first valid index of the longest series.
    end_point : int, float, str, or None, default=None
        The ending index/date for the data to be plotted.
    logscale : bool, default=False
        If True, a logarithmic scale is used for the y-axis.
    show_mean : bool, default=False
        If True, a dotted horizontal line representing the mean of
        each series is added.
    show_bands : dict or bool, default=False
        If True or a dictionary, rolling standard deviation bands are
        added. A dictionary can specify "window" (default 20),
        "std" (default 2) (example:  {"window": 260, "std": 2}).
    show_moving_average : int or bool, default=False
        If True or an integer, a moving average line is added. If True,
        a default window of 260 is used. If an integer, it defines the
        rolling window size.
    show_rangeselector : bool, default=True
        If True and the index is a datetime type, a range selector
        (1M, 6M, 1Y, All) is added to the x-axis.
    plotly_template : str or None, default="plotly_dark"
        The Plotly template to use for the chart styling.
    chart_width : int or None, default=None
        The width of the chart in pixels.
    chart_height : int or None, default=None
        The height of the chart in pixels.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The generated interactive Plotly line chart.
    """

    # Convert Series to DataFrame if needed
    data_df = data.to_frame() if isinstance(data, pd.Series) else data.copy()

    # Auto-detect y_col: pick first numeric column (excluding x_col)
    if y_column is None:
        num_cols = [c for c in data_df.select_dtypes(include="number").columns]
        y_column = num_cols
    else:
        if isinstance(y_column, str):
            y_column = [y_column]


    # Identify the longest time series and its starting point
    # Count non-null observations for each column
    series_lengths = data_df[y_column].count()

    # Identify the longest series
    longest_series_name = series_lengths.idxmax()

    # Get the corresponding series
    longest_series = data_df[longest_series_name]

    # Get its first valid index
    longest_series_start = longest_series.first_valid_index()

    # Use automatically detected start point if none is provided
    if start_point is None:
        start_point = longest_series_start

    if show_moving_average:
        window = 260 if isinstance(show_moving_average, bool) else show_moving_average

        # Calculate the rolling mean for each column based on the window
        mvg_avg_data_dict = {}
        for column in y_column:
            mvg_avg_data = data_df[column].rolling(window=window).mean().loc[start_point:end_point]
            mvg_avg_data_dict[column] = mvg_avg_data


    if show_bands:
        # Allow configuration via dictionary
        window = show_bands.get("window", 20) if isinstance(show_bands, dict) else 20
        std_multiplier = show_bands.get("std", 2) if isinstance(show_bands, dict) else 2
        alpha = 0.5 # s

        band_data_dict = {}
        for column in y_column:

            # Calculate rolling mean and standard deviation
            rolling_std = data_df[column].rolling(window=window).std()

            upper_band = (data_df[column] + std_multiplier * rolling_std).loc[start_point:end_point]
            lower_band = (data_df[column] - std_multiplier * rolling_std).loc[start_point:end_point]
            band_data_dict[column] = (upper_band, lower_band)



    # Adjust the length of the df based on start_point and end_point if available
    if end_point is not None:
        data_df = data_df.loc[start_point:end_point]
    else:
        data_df = data_df.loc[start_point:]


    fig = px.line(
        data_frame = data_df,
        title = chart_title or f"{y_column}",
        subtitle = chart_subtitle or None,
        y = y_column,
        log_y=logscale,
        template=plotly_template,
        width = chart_width,
        height = chart_height,
    )

    # 2. Optional mean line
    if show_mean:
        for column in y_column:
            mean_val = data_df[column].mean()

            fig.add_trace(
                go.Scatter(
                    x=data_df.index,
                    y=[mean_val] * len(data_df),
                    mode="lines",
                    line=dict(
                        color="#94A3B8",
                        dash="dot",
                        width=1.5
                    ),
                    name=f"Mean {column}: {mean_val:.2f}",
                    showlegend=True,
                )
            )

    if show_moving_average:
        for column in y_column:
            mvg_avg_data = mvg_avg_data_dict[column]


            # Get original column colour
            original_color = next(
            trace.line.color for trace in fig.data if trace.name == column
            )
            fig.add_trace(
                go.Scatter(
                    x=data_df.index,
                    y=mvg_avg_data,
                    mode="lines",
                    line=dict(
                        color=original_color,
                        dash="dot",
                        width=1.5
                    ),
                    opacity=0.8,
                    name=f"{column} (Moving Average)",
                    showlegend=True,
                )
            )

    if show_bands:
        for column in y_column:
            upper_band, lower_band = band_data_dict[column]

            # Upper band
            fig.add_trace(go.Scatter(
                x=data_df.index,
                y=upper_band,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Lower band + fill between upper and lower
            fig.add_trace(go.Scatter(
                x=data_df.index,
                y=lower_band,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(100, 100, 100, {alpha})",
                showlegend=False,
                hoverinfo="skip"
            ))
            
    if show_rangeselector and pd.api.types.is_datetime64_any_dtype(data_df):
        fig.update_xaxes(
            rangeselector=dict(
                bgcolor="rgba(30, 41, 59, 0.8)",
                activecolor="#38BDF8",
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            )
        )
    return fig

