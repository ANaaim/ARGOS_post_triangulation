from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output

# ============================================================
# LOAD DATA
# ============================================================

path_to_data = Path(r".\data\pre_processed\absolute_position_errors.parquet")

df = pl.read_parquet(path_to_data)

POINT_ORDER = [
    "C7",
    "IJ",
    "T10",
    "LS",
    "GH",
    "EJC",
    "EM",
    "EL",
    "WJC",
    "US",
    "RS",
    "HM2",
    "HM5",
]


# ============================================================
# YOUR PLOT FUNCTION
# ============================================================


def plot_error_quantiles(
    df: pl.DataFrame,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    points: list[str] | None = None,
    models: list[str] | None = None,
    coordinate_system: str = "anatomical",
):

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    df_plot = df.filter(pl.col("coordinate_system") == coordinate_system)

    if subjects is not None:
        df_plot = df_plot.filter(pl.col("subject").is_in(subjects))

    if tasks is not None:
        df_plot = df_plot.filter(pl.col("task").is_in(tasks))

    if points is not None:
        df_plot = df_plot.filter(pl.col("point_short").is_in(points))

    if models is not None:
        df_plot = df_plot.filter(pl.col("model").is_in(models))

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    error_columns = [
        "error_x",
        "error_y",
        "error_z",
        "error_norm",
    ]

    quantiles = (
        df_plot.unpivot(
            index=["point_short", "model"],
            on=error_columns,
            variable_name="component",
            value_name="error",
        )
        .drop_nulls("error")
        .group_by(["point_short", "model", "component"])
        .agg(
            pl.col("error").quantile(0.05).alias("q05"),
            pl.col("error").quantile(0.20).alias("q20"),
            pl.col("error").quantile(0.50).alias("median"),
            pl.col("error").quantile(0.70).alias("q70"),
            pl.col("error").quantile(0.95).alias("q95"),
            pl.len().alias("n"),
        )
        .sort(["point_short", "model", "component"])
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    components = [
        "error_x",
        "error_y",
        "error_z",
        "error_norm",
    ]

    titles = [
        "X error",
        "Y error",
        "Z error",
        "Error norm",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
    )

    positions = {
        "error_x": (1, 1),
        "error_y": (1, 2),
        "error_z": (2, 1),
        "error_norm": (2, 2),
    }

    model_colors = {
        "SynthPose": "#1f77b4",
        "RTMPose": "#ff7f0e",
    }

    available_models = quantiles.select("model").unique().to_series().to_list()

    available_points = quantiles.select("point_short").unique().sort("point_short").to_series().to_list()

    # --------------------------------------------------------
    # Draw custom percentile boxes
    # --------------------------------------------------------

    for component in components:

        row, col = positions[component]

        q_component = quantiles.filter(pl.col("component") == component)

        for model_index, model in enumerate(available_models):

            q_model = q_component.filter(pl.col("model") == model)

            color = model_colors.get(model, "#636EFA")

            # Offset models slightly so they do not overlap
            offset = (model_index - (len(available_models) - 1) / 2) * 0.25

            for point_index, point in enumerate(available_points):

                q = q_model.filter(pl.col("point_short") == point)

                if q.height == 0:
                    continue

                q05 = q["q05"][0]
                q20 = q["q20"][0]
                median = q["median"][0]
                q70 = q["q70"][0]
                q95 = q["q95"][0]
                n = q["n"][0]

                x = point_index + offset

                # --------------------------------------------
                # 5 -> 95 percentile whisker
                # --------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[q05, q95],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=2,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Box 20 -> 70 percentile
                # --------------------------------------------

                fig.add_shape(
                    type="rect",
                    x0=x - 0.09,
                    x1=x + 0.09,
                    y0=q20,
                    y1=q70,
                    line=dict(
                        color=color,
                        width=2,
                    ),
                    fillcolor=color,
                    opacity=0.25,
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Median
                # --------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=[x - 0.09, x + 0.09],
                        y=[median, median],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=3,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

                # --------------------------------------------
                # Hover point
                # --------------------------------------------

                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[median],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=8,
                        ),
                        name=model,
                        legendgroup=model,
                        showlegend=(component == "error_x" and point_index == 0),
                        customdata=[
                            [
                                point,
                                model,
                                q05,
                                q20,
                                median,
                                q70,
                                q95,
                                n,
                            ]
                        ],
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Model: %{customdata[1]}<br>"
                            "5%: %{customdata[2\]:.2f}<br>"
                            "20%: %{customdata[3\]:.2f}<br>"
                            "Median: %{customdata[4\]:.2f}<br>"
                            "70%: %{customdata[5\]:.2f}<br>"
                            "95%: %{customdata[6\]:.2f}<br>"
                            "n: %{customdata[7]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
    # -----------------------------------------
    # X axis labels
    # --------------------------------------------------------
    points_in_data = set(quantiles.get_column("point_short").unique().to_list())
    available_points = [point for point in POINT_ORDER if point in points_in_data]

    tickvals = list(range(len(available_points)))

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=available_points,
                tickangle=45,
                row=row,
                col=col,
            )

            fig.update_yaxes(
                title_text="Error (mm)",
                row=row,
                col=col,
            )

    fig.update_layout(
        height=850,
        template="plotly_white",
        title="Markerless position errors",
        hovermode="closest",
        margin=dict(
            l=60,
            r=30,
            t=80,
            b=100,
        ),
    )

    return fig


# ============================================================
# VALUES AVAILABLE IN DATASET
# ============================================================

all_tasks = df.select("task").unique().sort("task").to_series().to_list()

all_subjects = df.select("subject").unique().sort("subject").to_series().to_list()

all_points = df.select("point_short").unique().sort("point_short").to_series().to_list()

all_models = df.select("model").unique().sort("model").to_series().to_list()


# ============================================================
# DASH APPLICATION
# ============================================================

app = Dash(__name__)
points_in_data = set(df.get_column("point_short").unique().to_list())
all_points = [point for point in POINT_ORDER if point in points_in_data]

# ============================================================
# DASH LAYOUT
# ============================================================

app.layout = html.Div(
    [
        html.H1(
            "Markerless Position Error Analysis",
            style={"textAlign": "center"},
        ),
        # ----------------------------------------------------
        # TASKS
        # ----------------------------------------------------
        html.H3("Tasks"),
        dcc.Checklist(
            id="task-selector",
            options=[
                {
                    "label": task,
                    "value": task,
                }
                for task in all_tasks
            ],
            value=all_tasks,
            inline=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # SUBJECTS
        # ----------------------------------------------------
        html.H3("Subjects"),
        dcc.Dropdown(
            id="subject-selector",
            options=[
                {
                    "label": subject,
                    "value": subject,
                }
                for subject in all_subjects
            ],
            value=all_subjects,
            multi=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------
        html.H3("Points"),
        dcc.Dropdown(
            id="point-selector",
            options=[
                {
                    "label": point,
                    "value": point,
                }
                for point in all_points
            ],
            value=all_points,
            multi=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------
        html.H3("Models"),
        dcc.Checklist(
            id="model-selector",
            options=[
                {
                    "label": model,
                    "value": model,
                }
                for model in all_models
            ],
            value=all_models,
            inline=True,
        ),
        html.Hr(),
        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------
        dcc.Loading(
            type="circle",
            children=dcc.Graph(
                id="error-graph",
                style={"height": "900px"},
            ),
        ),
    ],
    style={
        "width": "95%",
        "margin": "auto",
        "fontFamily": "Arial",
    },
)


# ============================================================
# CALLBACK
# ============================================================


@app.callback(
    Output("error-graph", "figure"),
    Input("task-selector", "value"),
    Input("subject-selector", "value"),
    Input("point-selector", "value"),
    Input("model-selector", "value"),
)
def update_graph(
    selected_tasks,
    selected_subjects,
    selected_points,
    selected_models,
):

    return plot_error_quantiles(
        df=df,
        tasks=selected_tasks,
        subjects=selected_subjects,
        points=selected_points,
        models=selected_models,
        coordinate_system="anatomical",
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
