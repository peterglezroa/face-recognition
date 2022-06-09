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
    df["vgg"] = df["variable"].map(lambda x: True if "vgg" in x else False)
    df["resnet"] = df["variable"].map(lambda x: True if "vgg" in x else False)
    df["senet"] = df["variable"].map(lambda x: True if "vgg" in x else False)
    df = df.rename(columns={"value": "NMI", "variable": "Modelos"})

    # Remove rows with empty values
    df.dropna(subset=["NMI"], inplace=True)

    if args.folder is not None:
        df.to_csv(os.path.join(args.folder, "melted_df.csv"))

    # Melt to analyse agglomerative vs kmeans
    ax = sns.boxplot(data=df, x="cluster alg", y="NMI", showfliers=False)
    if args.folder is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.figure.savefig(os.path.join(args.folder, "cluster_comparison.png"))

    # Stop analysing multiview
    df = df[df["multiview"] != "MVC"]

    # Dataset comparison
    ax = sns.catplot(kind="box", data=df, x="multiview",
        y="NMI", col="Dataset", col_wrap=2, showfliers=False)
    ax.set_axis_labels("Acercamientos Propuestos en esta Investigación")
    if args.folder is not None:
#        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.fig.savefig(os.path.join(args.folder, "dataset_comparison.png"))

    # Dataset comparison in celeba dataset
    ax = sns.catplot(kind="box", data=df[(df["Dataset"]=="celeba") & (df["multiview"]!="MVEC")],
        x="multiview", y="NMI", col="vgg", col_wrap=2, showfliers=False)
    ax.set_axis_labels("Acercamientos Propuestos en esta Investigación")
    if args.folder is not None:
#        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.fig.savefig(os.path.join(args.folder, "celeba_vgg.png"))
    
    # Model score per dataset without proyect
    df["Modelos"] = df["Modelos"].str.split(' ').str[1]
    ax = sns.catplot(kind="box", data=df[df["multiview"]=="S"],
        x="Modelos", y="NMI", col="Dataset", col_wrap=2, showfliers=False)
    ax.set_axis_labels("Acercamientos Propuestos en esta Investigación")
    if args.folder is not None:
#        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.fig.savefig(os.path.join(args.folder, "singles.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run S, CC, MVEC and MVSC models tests for an specific dataset"
    )
    parser.add_argument("-f", "--folder", type=str,
        help="The folder in which the images are going to be saved.")

    main(parser.parse_args())
