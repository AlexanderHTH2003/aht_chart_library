import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import numpy as np


def aht_scatter_chart(
    data: pd.DataFrame | pd.Series,
    x_column: str | None = None,
    y_column: str | None = None,
    chart_title: str | None = None,
    start_point: int | float | str | None = None,
    end_point: int | float | str | None = None,
    show_trendline: bool = False,
    show_confidence_interval: bool | dict = False,
    show_means: bool = False,
    show_extrema: bool = False,
    show_latest: bool = False,
    show_index: bool = False,
    plotly_template: str = "plotly_dark",
):
    """
    Generate an interactive scatter plot using Plotly with optional statistical overlays.

    This function automatically detects numerical columns if none are provided and 
    supports various enhancements including OLS trendlines, confidence intervals, 
    mean lines, and extrema markers.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The input data containing the variables to plot.
    x_column : str, optional
        The name of the column to use for the X-axis. If None, it will be auto-detected 
        or default to the DataFrame index.
    y_column : str, optional
        The name of the column to use for the Y-axis. If None, it will be auto-detected.
    chart_title : str, optional
        The title of the chart. Defaults to "{y_column} vs. {x_column}".
    start_point : int, float, str, optional
        The starting index/label to slice the data from.
    end_point : int, float, str, optional
        The ending index/label to slice the data to.
    show_trendline : bool, default False
        If True, plots an Ordinary Least Squares (OLS) regression line.
    show_confidence_interval : bool or dict, default False
        If True or a dict, plots a shaded confidence interval around the trendline. 
        Can accept a dictionary with keys 'window' (for HAC covariance) and 'confidence_level'.
    show_means : bool, default False
        If True, adds horizontal and vertical dotted lines representing the axes means.
    show_extrema : bool, default False
        If True, highlights the maximum and minimum Y-values with distinct markers.
    show_latest : bool, default False
        If True, highlights the most recent valid observation.
    show_index : bool, default False
        If True, includes the index value in the hover tooltip.
    plotly_template : str, default "plotly_dark"
        The Plotly theme template to apply to the figure.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The resulting interactive Plotly scatter plot figure.
    """
    data_df = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    num_cols = list(data_df.select_dtypes(include="number").columns)


    # Auto-detect x_column and y_column if omitted
    if x_column is None and y_column is None:
        if len(num_cols) >= 2:
            x_column, y_column = num_cols[0], num_cols[1]
        else:
            y_column = num_cols[0] if num_cols else data_df.columns[0]
            x_column = data_df.index.name or "Index"
    elif x_column is None:
        x_column = data_df.index.name or "Index"
    else:
        if y_column is None:
            rem_cols = [c for c in num_cols if c != x_column]
            y_column = rem_cols[0] if rem_cols else data_df.columns[0]

    # Adjust the length of the df based on start_point and end_point if available
    if start_point is not None and end_point is not None:
        data_df = data_df.loc[start_point:end_point]
    elif start_point is not None:
        data_df = data_df.loc[start_point:]
    elif end_point is not None:
        data_df = data_df.loc[:end_point]

    # Parse confidence interval configuration
    ci_enabled = bool(show_confidence_interval)

    if isinstance(show_confidence_interval, dict):
        hac_window = show_confidence_interval.get("window")
        confidence_level = float(show_confidence_interval.get("confidence_level", 0.95))

    # Base px.scatter plot
    fig = px.scatter(
        data_df,
        x=x_column,
        y=y_column,
        title=chart_title or f"{y_column} vs. {x_column}",
        opacity=0.75,
        template=plotly_template,
    )

    # Optional OLS regression line & confidence interval band
    if (show_trendline or ci_enabled) and pd.api.types.is_numeric_dtype(data_df[x_column]):
        valid_values = pd.DataFrame(
            {"x": data_df[x_column].astype(float), "y": data_df[y_column].astype(float)}
        ).dropna()

        if len(valid_values) >= 3:
            X_obs = sm.add_constant(valid_values["x"].to_numpy(), has_constant="add")
            y_obs = valid_values["y"].to_numpy()

            # Use Newey-West HAC covariance if an autocorrelation window is provided
            if hac_window is not None and int(hac_window) > 0:
                maxlags = min(int(hac_window), len(valid_values) - 2)
                ols_res = sm.OLS(y_obs, X_obs).fit(
                    cov_type="HAC", cov_kwds={"maxlags": maxlags}
                )
            else:
                ols_res = sm.OLS(y_obs, X_obs).fit()

            # Dense, sorted X-grid for smooth confidence curves
            x_grid = np.linspace(valid_values["x"].min(), valid_values["x"].max(), 200)
            X_grid = sm.add_constant(x_grid, has_constant="add")
            pred_df = ols_res.get_prediction(X_grid).summary_frame(
                alpha=1.0 - confidence_level
            )

            # Draw shaded CI band first so the OLS line sits clearly in front
            if ci_enabled:
                fig.add_scatter(
                    x=x_grid,
                    y=pred_df["mean_ci_upper"],
                    mode="lines",
                    line=dict(color="rgba(124, 58, 237, 0.4)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
                fig.add_scatter(
                    x=x_grid,
                    y=pred_df["mean_ci_lower"],
                    mode="lines",
                    line=dict(color="rgba(124, 58, 237, 0.4)", width=1),
                    fill="tonexty",
                    fillcolor="rgba(124, 58, 237, 0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                )

            # Draw OLS regression line on top of the band
            if show_trendline:
                r_val = valid_values["x"].corr(valid_values["y"])
                r2_val = ols_res.rsquared
                fig.add_scatter(
                    x=x_grid,
                    y=pred_df["mean"],
                    mode="lines",
                    name=f"OLS (r={r_val:.2f}, R²={r2_val:.2f})",
                    line=dict(color="#7C3AED", width=2.5, dash="dash"),
                )

    if show_index:
        index_label = data_df.index.name or "Index"
        hovertemplate = (
            f"<b>{index_label}:</b> %{{customdata[0]}}<br>"
            f"<b>{x_column}:</b> %{{x}}<br>"
            f"<b>{y_column}:</b> %{{y}}"
            "<extra></extra>"
        )
    else:
        hovertemplate = (
            f"<b>{x_column}:</b> %{{x}}<br>"
            f"<b>{y_column}:</b> %{{y}}"
            "<extra></extra>"
        )

    # Optional X and Y mean lines (4-quadrant split)
    if show_means:
        y_mean = data_df[y_column].mean()
        fig.add_hline(
            y=y_mean,
            line_dash="dot",
            line_color="#94A3B8",
            annotation_text=f"X-Axis Avg: {y_mean:.2f}",
            annotation_position="top left",
        )
        if pd.api.types.is_numeric_dtype(data_df[x_column]):
            x_mean = pd.Series(data_df[x_column]).mean()
            fig.add_vline(
                x=x_mean,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text=f"Y-Axis Avg: {x_mean:.2f}",
                annotation_position="top right",
            )


    # adjustments
    fig.update_traces(
        customdata=np.stack([data_df.index.astype(str)], axis=-1),
        hovertemplate=hovertemplate,
        marker=dict(size=4, color="#0D9488",),
        selector=dict(mode="markers"),

    )

    fig.update_layout(
        template=plotly_template,
        font=dict(family="Times New Roman", size=12),
        hovermode="closest",
        hoverlabel=dict(
            bordercolor="#334155",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Optional Min / Max Y markers
    if show_extrema and not data_df[y_column].dropna().empty:
        s_clean = data_df[y_column].dropna()
        id_max = s_clean.idxmax()
        id_min = s_clean.idxmin()
        
        pos_max = data_df.index.get_loc(id_max)
        pos_min = data_df.index.get_loc(id_min)
        x_vals = pd.Series(data_df[x_column])

        fig.add_scatter(
            x=[x_vals.iloc[pos_max], x_vals.iloc[pos_min]],
            y=[data_df[y_column].iloc[pos_max], data_df[y_column].iloc[pos_min]],
            mode="markers",
            name="Max / Min",
            marker=dict(
                color=["#2FD440", "#FF2626"],
                size=6,
                symbol="diamond",
                line=dict(width=5, color="#E300F4"),
            ),
            customdata=np.stack([[str(id_max), str(id_min)]], axis=-1),
            hovertemplate=hovertemplate,
        )

    # Optional Latest Value marker
    if show_latest and not data_df[y_column].dropna().empty:
        last_valid = data_df[y_column].last_valid_index()
        last_value = data_df.loc[last_valid, y_column]
        last_x = data_df.loc[last_valid, x_column]

        fig.add_scatter(
            x=[last_x],
            y=[last_value],
            mode="markers+text",
            name="Latest Value",
            textfont=dict(size=12, color="#F59E0B"),
            marker=dict(
                color="#FFA200",
                size=11,
                symbol="circle",
            ),
            customdata=np.stack([[str(last_valid)]], axis=-1),
            hovertemplate=hovertemplate,
        )

    return fig