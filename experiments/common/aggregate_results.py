import numpy as np
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import pandas as pd

from common.cd_diagrams import draw_cd_diagram

try:
    from Orange.evaluation import compute_CD, graph_ranks
except:
    print("could not import Orange. Try with: pip install orange3==3.30")
    print("skipping CD diagram creation")
    compute_CD = None
    graph_ranks = None
    pass
def save_support_images(resultsdir, support_input, support_target):
    os.makedirs(resultsdir, exist_ok=True)

    for idx, (image, target) in enumerate(zip(support_input, support_target)):
        rgb = equalize_hist(image.numpy())
        fig, ax = plt.subplots()
        ax.imshow(rgb.transpose(1, 2, 0))
        ax.axis("off")
        fig.savefig(os.path.join(resultsdir, f"{idx}-{target.replace(' ','_')}.png"),
                    pad_inches=0, bbox_inches="tight")
        plt.close(fig)


def save_results(outputfolder, y_pred, query_target, y_score, classes):
    os.makedirs(outputfolder, exist_ok=True)
    results_dict = classification_report(y_pred=y_pred, y_true=query_target, target_names=classes, output_dict=True)
    pd.DataFrame(results_dict).T.to_csv(os.path.join(outputfolder,"results.csv"))
    np.savez(os.path.join(outputfolder, "predictions.npz"),
             y_pred=y_pred,
             y_score=y_score.numpy(),
             classes=classes,
             query_target=query_target
             )
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes),
          file=open(os.path.join(outputfolder, "results.txt"), "w"))


def write_and_aggregate_results(datadir, resultsdir, datasets, compare_models=None):
    os.makedirs(resultsdir, exist_ok=True)

    stats = []
    #datasets = [d for d in os.listdir(resultsdir) if os.path.isdir(os.path.join(resultsdir, d))]
    for dataset in datasets:
        models = [f for f in os.listdir(os.path.join(datadir, dataset)) if
                  os.path.isdir(os.path.join(datadir, dataset, f))]

        if compare_models is not None:
            models = [m for m in models if m in compare_models]

        for model in models:
            resultsfile = os.path.join(datadir, dataset, model, "results.csv")
            if os.path.exists(resultsfile):
                accuracy = pd.read_csv(resultsfile, index_col=0).loc["accuracy"].iloc[0]
                stats.append({
                    "dataset": dataset,
                    "model": model,
                    "accuracy": accuracy
                })

    df = pd.DataFrame(stats)
    df = df.pivot("model", "dataset", "accuracy")

    df = df * 100

    stacked_df = df.stack().reset_index()
    stacked_df.columns = ["classifier_name", "dataset_name", "accuracy"]
    p_values, average_ranks = draw_cd_diagram(df_perf=stacked_df, title='average rank by accuracy', labels=True,
                                              savepath=os.path.join(resultsdir, "cd-diagram.png"))

    if compute_CD is not None and len(average_ranks) <= 20: # Orange only supports up to 20 classifiers
        cd = compute_CD(average_ranks, len(datasets), alpha="0.1")
        graph_ranks(average_ranks,
                    names=average_ranks.index,
                    cd=cd,
                    width=5,
                    textspace=1.5,
                    reverse=True)
        plt.savefig(os.path.join(resultsdir, "cd-diagram_orange.pdf"))
    else:
        print("skipping generation of orange CD diagrams because of import error!")

    # get a list of datasets and models before adding more columns
    models = list(df.index)
    datasets = list(df.columns)

    # test significance of bagofmaml vs all others
    reference_model = "meteor"
    df = add_wilcoxon_test(df, method=reference_model, comparisons=[d for d in models if d != reference_model])

    # add average rank
    df.insert(0, "avg_ranks", average_ranks)
    df = df.sort_values(by="avg_ranks")
    #df = add_avg_rank(df, datasets)

    df.to_csv(os.path.join(resultsdir, "results.csv"))


    fname = os.path.join(resultsdir,"results.tex")
    print(df.drop("wcx-test", axis=1).drop("wcx-pvalue", axis=1).to_latex(float_format="%.1f"), file=open(fname,"w"))

    print(df.to_markdown(floatfmt=".1f"))
    fname = os.path.join(resultsdir, "results.md")
    print(df.to_markdown(floatfmt=".1f"), file=open(fname,"w"))


def add_wilcoxon_test(df, method, comparisons):

    # make a copy before adding new columns
    df_copy = df.copy()
    reference = df_copy.loc[method]

    # insert empty
    df.insert(0, "wcx-pvalue", None)
    df.insert(0, "wcx-test", None)

    for competitor in comparisons:
        comparison = df_copy.loc[competitor]

        #result, z_statistic = wilcoxon_signed_ranks_test(reference, comparison)

        from scipy.stats import wilcoxon
        rs = wilcoxon(reference - comparison, alternative="two-sided")

        sig = ""
        if rs.pvalue < 0.1: # reject null hypothesis: there is a difference between classifiers
            sig += "*"
        if rs.pvalue < 0.05: # reject null hypothesis: there is a difference between classifiers
            sig += "*" # make **

        df.loc[competitor, "wcx-test"] = sig # reject null hypothesis: there is a difference between classifiers
        df.loc[competitor, "wcx-pvalue"] = rs.pvalue

    return df


def wilcoxon_signed_ranks_test(comparison, reference):
    """
    A statistical test if the average difference of two classifiers is significantly different from zero.

    Hypothesis: reference and comparison are not equally accurate
    Null-Hypothesis: reference and comparison are equally accurate

    returns True/False whether the Hypothesis is True or False

    implementation following:  WILCOXON SIGNED-RANKS TEST
    of https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
    """
    d = reference - comparison
    abs_d = np.abs(d)
    ranks = np.argsort(abs_d) + 1

    R_plus = ranks[d > 0].sum() + 0.5 * ranks[d == 0].sum()
    R_minus = ranks[d < 0].sum() + 0.5 * ranks[d == 0].sum()

    T = np.min([R_plus, R_minus])
    N = len(reference)

    print(R_plus, R_minus)

    z = (T - 0.25*N*(N+1)) / np.sqrt( (1./24.)*N * (N+1) * (2 * N + 1) )

    return z < -1.96, z # "With α = 0.05, the null-hypothesis can be rejected if z is smaller than −1.96"



def add_avg_rank(df, datasets):
    winners = []
    for dataset in datasets:
        winners.append(list(df[dataset].sort_values(ascending=False).index))
    winners = np.array(winners)

    won_datasets = []
    for model in df.index:
        won_datasets.append((winners == model).sum(0))
    won_datasets = np.array(won_datasets)

    avg_rank = np.matmul(won_datasets, np.arange(1, won_datasets.shape[0] + 1)) / won_datasets.shape[0]
    df.insert(0, "avg_rank", avg_rank)
    return df
