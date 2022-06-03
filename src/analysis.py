"""
analysis.py
-----------
Script to analyze the results from the run_tests script
"""
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

RESULTS_PATH = "../results.csv"

def main(args):
    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    base_df = pd.read_csv(RESULTS_PATH)

    # Drop Date col
    base_df = base_df.drop(columns=["Date"])

    # Define types of cols
    id_vars = ["Dataset", "# People"]
    time_cols = [col for col in base_df if col.endswith("time")]

    # Melt for better analysis
    df = base_df.drop(columns=time_cols)
    val_vars = [col for col in df if col not in id_vars]
    df = base_df.melt(id_vars=id_vars, value_vars=val_vars)
    df["cluster alg"] = df["variable"].str.split(' ').str[-1]
    df["multiview"] = df["variable"].str.split(' ').str[0]

    # Remove rows with empty values
    df.dropna(subset=["value"], inplace=True)

    if args.folder is not None:
        df.to_csv(os.path.join(args.folder, "melted_df.csv"))

    # Melt to analyse agglomerative vs kmeans
    ax = sns.boxplot(data=df, x="cluster alg", y="value")
    if args.folder is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.figure.savefig(os.path.join(args.folder, "cluster_alg_boxplot.png"))

    # MVC clustering algorithms comparison
    ax = sns.catplot(kind="box", data=df[df["multiview"] != "MVC"], x="multiview",
        y="value", col="Dataset", col_wrap=2)
    if args.folder is not None:
#        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.fig.savefig(os.path.join(args.folder, "multiview_boxplot.png"))
    
    # Melt to analyse single, cc, mvc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run S, CC, MVEC and MVSC models tests for an specific dataset"
    )
    parser.add_argument("-f", "--folder", type=str,
        help="The folder in which the images are going to be saved.")

    main(parser.parse_args())
