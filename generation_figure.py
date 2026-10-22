from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

def all_subjects_tasks(subjects_to_process: list[str], tasks_to_process:list[str], dof_order: list[str],
                                path_to_data: Path,
                                output_dir: Path, data_to_plot: str,
                                limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):
    if not path_to_data.exists():
        raise FileNotFoundError(f"Data file not found: {path_to_data}")
    if path_to_data.suffix != ".parquet":
        raise ValueError(f"Data file must be a parquet file: {path_to_data}")
    
    df = pl.read_parquet(path_to_data)
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

            #
            # BOXPLOT
            #
            plt.figure(figsize=(18, 8))

            sns.boxplot(
                data=plot_df,
                x="dof",
                y=data_to_plot,
                hue="model",
                order=dof_order,
                showfliers=False,
                whis=limit_boxplot_whiskers,
            )

            plt.title(
                f"{subject} - {task}",
                fontsize=14,
                fontweight="bold",
            )

            plt.ylabel(
                "Absolute angular error (deg)"
            )

            plt.xlabel(
                "Degree of Freedom"
            )

            plt.xticks(rotation=45)

            plt.tight_layout()
            figure_name=f"{subject}_{task}_boxplot.png"
            plt.savefig(
                out_dir / figure_name,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()


def by_task(tasks_to_process:list[str], name_tasks_order: dict[str, str],
        dof_order: list[str], name_dof_order: dict[str, str],
        path_to_data: Path,
        output_dir: Path, data_to_plot: str,
        limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):

    if not path_to_data.exists():
        raise FileNotFoundError(f"Data file not found: {path_to_data}")
    if path_to_data.suffix != ".parquet":
        raise ValueError(f"Data file must be a parquet file: {path_to_data}")
    df = pl.read_parquet(path_to_data)
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

        #
        # BOXPLOT
        #
        plt.figure(figsize=(18, 8))

        sns.boxplot(
            data=plot_df,
            x="dof",
            y=data_to_plot,
            hue="model",
            order=dof_order,
            showfliers=False,
            whis=limit_boxplot_whiskers,
        )

        plt.title(
            f"{name_tasks_order[task]} - All subjects",
            fontsize=14,
            fontweight="bold",
        )

        plt.ylabel("Absolute angular error (deg)")
        plt.xlabel("Degree of Freedom")
        ax = plt.gca()

        ax.set_xticklabels(
            [name_dof_order[d] for d in dof_order],
            rotation=30,
            ha="right",
        )

        plt.tight_layout()
        figure_name=f"{task}_boxplot.png"
        plt.savefig(
            out_dir / figure_name,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

def by_dof(tasks_to_process:list[str], name_tasks_order: dict[str, str],
        dof_order: list[str], name_dof_order: dict[str, str],
        path_to_data: Path, 
        output_dir: Path, data_to_plot: str,
        limit_boxplot_whiskers: tuple[float, float] = (2.5, 97.5)):

    if not path_to_data.exists():
        raise FileNotFoundError(f"Data file not found: {path_to_data}")
    if path_to_data.suffix != ".parquet":
        raise ValueError(f"Data file must be a parquet file: {path_to_data}")
    df = pl.read_parquet(path_to_data)
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

        #
        # BOXPLOT
        #
        plt.figure(figsize=(18, 8))

        sns.boxplot(
            data=plot_df,
            x="task",
            y=data_to_plot,
            hue="model",
            order=tasks_to_process,
            showfliers=False,
            whis=limit_boxplot_whiskers,
        )

        plt.title(
            f"{name_dof_order[dof]} - All subjects",
            fontsize=14,
            fontweight="bold",
        )

        plt.ylabel("Absolute angular error (deg)")
        plt.xlabel("Task")
        ax = plt.gca()

        ax.set_xticklabels(
            [name_tasks_order[d] for d in tasks_to_process],
            rotation=30,
            ha="right",
        )

        plt.tight_layout()
        figure_name=f"{dof}_boxplot.png"
        plt.savefig(
            out_dir / figure_name,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()



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
    df = pl.read_parquet(path_to_data)
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