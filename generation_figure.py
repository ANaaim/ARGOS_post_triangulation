from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

def load_dataframe(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix != ".parquet":
        raise ValueError(f"Expected a parquet file: {path}")

    return pl.read_parquet(path)


def save_boxplot(
    df,
    x,
    y,
    hue,
    order,
    name_order,
    title,
    xlabel,
    ylabel,
    output_file,
    whiskers=(2.5, 97.5),
):
    plt.figure(figsize=(18, 8))

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        showfliers=False,
        whis=whiskers,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if name_order is not None:
        ax = plt.gca()

        ax.set_xticklabels(
            [name_order[d] for d in order],
            rotation=0,
            #ha="right", could be set to "right" if needed
            )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def all_subjects_tasks(subjects_to_process: list[str], tasks_to_process:list[str], dof_order: list[str],
                                path_to_data: Path,
                                output_dir: Path, data_to_plot: str,
                                limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):
    df = load_dataframe(path_to_data)
    
    for subject in subjects_to_process:

        tasks = (
            df.filter(pl.col("subject") == subject)
            .select("task")
            .unique()
            .sort("task")
            .to_series()
            .to_list()
        )
        # remove tasks that are not in the tasks_to_process list
        tasks = [task for task in tasks if task in tasks_to_process]

        for task in tasks:

            print(subject, task)

            out_dir = output_dir/ subject 
            out_dir.mkdir(parents=True, exist_ok=True)

            plot_df = (
                df.filter(
                    (pl.col("subject") == subject)
                    & (pl.col("task") == task)
                )
                .to_pandas()
            )

            if len(plot_df) == 0:
                continue

            save_boxplot(
                df=plot_df,
                x="dof",
                y=data_to_plot,
                hue="model",
                order=dof_order,
                name_order = None,
                title=f"{subject} - {task}",
                xlabel="Degree of Freedom",
                ylabel="Absolute angular error (deg)",
                output_file=out_dir / f"{subject}_{task}_boxplot.png",
                whiskers=limit_boxplot_whiskers,
            )


def by_task(tasks_to_process:list[str], name_tasks_order: dict[str, str],
        dof_order: list[str], name_dof_order: dict[str, str],
        path_to_data: Path,
        output_dir: Path, data_to_plot: str,
        limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):

    df = load_dataframe(path_to_data)
    # Plot data per task for all subjects
    for task in tasks_to_process:

        print(task)

        out_dir = output_dir

        plot_df = (
            df.filter(pl.col("task") == task)
            .to_pandas()
        )

        if plot_df.empty:
            continue
    
        save_boxplot(
            df=plot_df,
            x="dof",
            y=data_to_plot,
            hue="model",
            order=dof_order,
            name_order = name_dof_order,
            title=f"{name_tasks_order[task]} - All subjects",
            xlabel="Degree of Freedom",
            ylabel="Absolute angular error (deg)",
            output_file=out_dir / f"{task}_boxplot.png",
            whiskers=limit_boxplot_whiskers,
        )


def by_dof(tasks_to_process:list[str], name_tasks_order: dict[str, str],
        dof_order: list[str], name_dof_order: dict[str, str],
        path_to_data: Path, 
        output_dir: Path, data_to_plot: str,
        limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):

    df = load_dataframe(path_to_data)
    # Plot data for each dof for all subjects
    for dof in dof_order:

        print(dof)

        out_dir = output_dir

        plot_df = (
            df.filter(pl.col("dof") == dof)
            .to_pandas()
        )

        if plot_df.empty:
            continue

        save_boxplot(
            df=plot_df,
            x="task",
            y=data_to_plot,
            hue="model",
            order=tasks_to_process,
            name_order = name_tasks_order,
            title=f"{name_dof_order[dof]} - All subjects",
            xlabel="Task",
            ylabel="Absolute angular error (deg)",
            output_file=out_dir / f"{dof}_boxplot.png",
            whiskers=limit_boxplot_whiskers,
        )





if __name__ == "__main__":

    data_to_plot = "error_deg" # abs_error_deg
    
    OUTPUT_DIR = Path("figures_strong_side") / data_to_plot / "boxplot"
    OUTPUT_DIR_TASK = OUTPUT_DIR / "task"
    OUTPUT_DIR_DOF = OUTPUT_DIR / "dof"


    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR_TASK.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR_DOF.mkdir(exist_ok=True, parents=True)

    path_to_data = Path ("kinematic_errors_strong_side.parquet")


    dof_order = [
        "Humerus_RotX",
        "Humerus_RotZ",
        "Humerus_RotY",
        "Forearm_1_RotZ",
        "Forearm_2_RotY",
    ]

    Name_dof_order = {
        "Humerus_RotX": "Shoulder Adduction-Abduction",
        "Humerus_RotZ": "Shoulder Flexion-Extension",
        "Humerus_RotY": "Shoulder Internal-External Rotation",
        "Forearm_1_RotZ": "Elbow Flexion-Extension",
        "Forearm_2_RotY": "Forearm Pronation-Supination",
    }

    tasks_order = ["01-eat-yaourt",
                "03-Open_a_Bottle_and_Pour",
                "04-Hand_to_Head",
                "13-playdoe",
                "16-comb-hair",
                "10-reach_and_grasp_far",
                "26-Cymbals"]

    Name_tasks_order = {
        "01-eat-yaourt": "Eat Yaourt",
        "03-Open_a_Bottle_and_Pour": "Open a Bottle and Pour",
        "04-Hand_to_Head": "Hand to Head",
        "13-playdoe": "Playdoe",
        "16-comb-hair": "Comb Hair",
        "10-reach_and_grasp_far": "Reach and Grasp Far",
        "26-Cymbals": "Cymbals",
    }
    df = load_dataframe(path_to_data)
    subjects_to_process = df.select("subject").unique().sort("subject").to_series().to_list()

    limit_boxplot_whiskers = (2.5, 97.5)
    
    all_subjects_tasks(
        subjects_to_process=subjects_to_process,
        tasks_to_process=tasks_order,
        dof_order=dof_order,
        path_to_data=path_to_data,
        output_dir=OUTPUT_DIR,
        data_to_plot=data_to_plot,
        limit_boxplot_whiskers=limit_boxplot_whiskers
    )
    by_task(tasks_to_process=tasks_order,name_tasks_order=Name_tasks_order,
        dof_order=dof_order,name_dof_order=Name_dof_order,
        path_to_data=path_to_data,
        output_dir=OUTPUT_DIR_TASK, data_to_plot=data_to_plot,
        limit_boxplot_whiskers=limit_boxplot_whiskers
    )

    by_dof(tasks_to_process=tasks_order,name_tasks_order=Name_tasks_order,
        dof_order=dof_order,name_dof_order=Name_dof_order,
        path_to_data=path_to_data,
        output_dir=OUTPUT_DIR_DOF, data_to_plot=data_to_plot,
        limit_boxplot_whiskers=limit_boxplot_whiskers
    )