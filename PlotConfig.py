from dataclasses import dataclass
import polars as pl

@dataclass
class PlotConfig:
    data_to_plot: str
    data_to_plot_label: str
    whiskers: tuple[float, float]
    dof_order: list[str] | None = None
    dof_labels: dict[str, str] | None = None
    task_order: list[str] | None = None
    task_labels: dict[str, str] | None = None

    def resolve(self, df: pl.DataFrame):

        if self.task_order is None:
            self.task_order = (
                df.select("task")
                .unique()
                .sort("task")
                .to_series()
                .to_list()
            )

        if self.dof_order is None:
            self.dof_order = (
                df.select("dof")
                .unique()
                .sort("dof")
                .to_series()
                .to_list()
            )

        if self.task_labels is None:
            self.task_labels = {
                task: task
                for task in self.task_order
            }

        if self.dof_labels is None:
            self.dof_labels = {
                dof: dof
                for dof in self.dof_order
            }

        if self.whiskers is None:
            self.whiskers = (0.0, 100.0)