from pathlib import Path

import polars as pl




def generate_excel_summary(df: pl.DataFrame, output_path: Path):
    """
    Génère un fichier Excel contenant un résumé statistique des erreurs angulaires absolues
    pour chaque combinaison de sujet, tâche, modèle et degré de liberté (DOF).
    """

    summary = (
        df.group_by(
            [
                "subject",
                "task",
                "model",
                "dof",
            ]
        )
        .agg(
            [
                pl.len().alias("n_frames"),
                pl.col("abs_error_deg").mean().alias("mean_abs_error"),
                pl.col("abs_error_deg").median().alias("median_abs_error"),
                pl.col("abs_error_deg").std().alias("std_abs_error"),
                pl.col("abs_error_deg").quantile(0.25).alias("q25"),
                pl.col("abs_error_deg").quantile(0.75).alias("q75"),
                pl.col("abs_error_deg").quantile(0.95).alias("p95"),
                pl.col("abs_error_deg").max().alias("max_abs_error"),
            ]
        )
        .sort(
            [
                "subject",
                "task",
                "dof",
                "model",
            ]
        )
    )

    summary.to_pandas().to_excel(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    
    df = pl.read_parquet("kinematic_errors_strong_side.parquet")
    output_path = Path("kinematic_errors_summary.xlsx")
    generate_excel_summary(df, output_path)