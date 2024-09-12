from pprint import pprint
import numpy as np
import scipy
import pandas as pd
import collections
from itertools import combinations
import bnlearn
import pooch
from cdt.data import load_dataset

from pywhy_graphs.functional import (
    make_graph_linear_gaussian,
    make_graph_multidomain,
    set_node_attributes_with_G,
    apply_linear_soft_intervention,
    sample_multidomain_lin_functions,
)
from pywhy_graphs.classes import AugmentedGraph
from pywhy_graphs.viz import draw

from dodiscover.cd import KernelCDTest
from dodiscover.ci import KernelCITest, FisherZCITest, Oracle, GSquareCITest
from dodiscover.constraint.skeleton import LearnMultiDomainSkeleton
from dodiscover.datasets import sample_from_graph

from dodiscover import PsiFCI, SFCI, Context, make_context, InterventionalContextBuilder


def resample_dataset(
    G,
    df,
    prior_multi_dist,
    nodes_to_resample,
    outcome_values,
    n_samples=1000,
    seed=12345,
):
    rng = np.random.default_rng(seed)

    new_df = np.zeros((n_samples, len(df.columns)))
    for idx in range(n_samples):
        row_idx = rng.integers(0, len(df))

        new_df[idx, :] = df.iloc[row_idx, :]

        for jdx, node in enumerate(nodes_to_resample):
            prior_dist = prior_multi_dist
            col_idx = np.argwhere(df.columns == node).squeeze()

            # sample which index from 1, 2, or 3 it hit
            new_sample_idx = rng.multinomial(1, pvals=prior_dist, size=1).squeeze()
            new_sample = outcome_values[np.argwhere(new_sample_idx == 1).squeeze()]
            new_df[idx, col_idx] = new_sample

            # print("new sample for ", node, new_sample)
            # sample the children according to a re-weighted Dirichlet distribution
            children = list(G.successors(node))
            for child in children:
                child_prior = prior_multi_dist.copy()
                child_prior[new_sample_idx] *= new_sample
                child_prior = rng.dirichlet(child_prior, 1)

                child_idx = np.argwhere(df.columns == child).squeeze()

                # sample which index from 1, 2, or 3 it hit for children
                child_sample_idx = rng.multinomial(1, pvals=child_prior, size=1).squeeze()
                child_sample = outcome_values[np.argwhere(child_sample_idx == 1).squeeze()]
                new_df[idx, child_idx] = child_sample
                # print("New sample for ", child, child_sample)

    new_df = pd.DataFrame(new_df)
    new_df.columns = df.columns
    return new_df


seed = 1234
n_jobs = -1
rng = np.random.default_rng(seed)
alpha = 0.05

# use pooch to download robustly from a url
url = "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz"
file_path = pooch.retrieve(
    url=url,
    known_hash="md5:39ee257f7eeb94cb60e6177cf80c9544",
)

df = pd.read_csv(file_path, delimiter=" ")

perturbations = [df.columns[perturbed_col] for perturbed_col in df["INT"].unique()]
n_proteins = len(df.columns) - 1

print(perturbations)

# the ground-truth dag is shown here: XXX: comment in when errors are fixed
ground_truth_dag = bnlearn.import_DAG("sachs", verbose=False)
ground_truth_G = ground_truth_dag["model"].to_directed()
G = draw(ground_truth_G, direction="TD", shape="circle")

# generate now bernoulli probability exogenous per protein
prior_protein_exp = rng.dirichlet(rng.standard_gamma(rng.integers(1, 4), size=3), 1).squeeze()

outcome_values = np.array([1, 2, 3])
nodes_to_resample = np.array(["Erk", "PKC", "PIP2"])

print(prior_protein_exp)
print(prior_protein_exp.sum(axis=0))

new_df = resample_dataset(
    ground_truth_G,
    df,
    prior_multi_dist=prior_protein_exp,
    nodes_to_resample=nodes_to_resample,
    outcome_values=outcome_values,
    n_samples=10000,
    seed=12345,
)

# %%
# Preprocess the dataset
# ----------------------
# Since the data is one dataframe, we need to process it into a form
# that is acceptable by dodiscover's :class:`constraint.PsiFCI` algorithm. We
# will form a list of separate dataframes.
unique_ints = df["INT"].unique()

# get the list of intervention targets and list of dataframe associated with each intervention
intervention_targets = []
data_cols = [col for col in df.columns if col != "INT"]
data = []
domain_ids = []
for interv_idx in unique_ints:
    _data = df[df["INT"] == interv_idx][data_cols].astype(int)
    data.append(_data)
    intervention_targets.append(df.columns[interv_idx])
    domain_ids.append(1)

    # append second domain
    _data = new_df[new_df["INT"] == interv_idx][data_cols].astype(int)
    data.append(_data)
    intervention_targets.append(df.columns[interv_idx])
    domain_ids.append(2)

print(len(data), len(intervention_targets), len(domain_ids))

# Setup constraint-based learner
# ------------------------------
# Since we have access to interventional data, the causal discovery algorithm
# we will use that leverages CI and CD tests to estimate causal constraints
# is the Psi-FCI algorithm :footcite:`Jaber2020causal`.

# Our dataset is comprised of discrete valued data, so we will utilize the
# G^2 (Chi-square) CI test.
ci_estimator = GSquareCITest(data_type="discrete")

# Since our data is entirely discrete, we can also use the G^2 test as our
# CD test.
cd_estimator = GSquareCITest(data_type="discrete")

alpha = 0.05
learner = SFCI(
    ci_estimator=ci_estimator,
    cd_estimator=cd_estimator,
    alpha=alpha,
    max_combinations=10,
    max_cond_set_size=4,
    n_jobs=-1,
)

# create context with information about the interventions
ctx_builder = make_context(create_using=InterventionalContextBuilder)
ctx: Context = ctx_builder.variables(data=data[0]).num_distributions(len(data)).build()

# %%
# Run the learning process
# ------------------------
# We have setup our causal context and causal discovery learner, so we will now
# run the algorithm using the :meth:`constraint.PsiFCI.fit` API, which is similar to scikit-learn's
# `fit` design. All fitted attributes contain an underscore at the end.
learner = learner.fit(
    data, ctx, domain_indices=domain_ids, intervention_targets=intervention_targets
)

# %%
# Analyze the results
# ===================
# Now that we have learned the graph, we will show it here. Note differences and similarities
# to the ground-truth DAG that is "assumed". Moreover, note that this reproduces Supplementary
# Figure 8 in :footcite:`Jaber2020causal`.
est_pag = learner.graph_

print(f"There are {len(est_pag.to_undirected().edges)} edges in the resulting PAG")

# %%
# Visualize the full graph including the F-node
# dot_graph = draw(est_pag, direction="LR")
# dot_graph.render(outfile="psi_pag_full.png", view=True, cleanup=True)

# %%
# Visualize the graph without the F-nodes
est_pag_no_fnodes = est_pag.subgraph(ctx.get_non_augmented_nodes())
dot_graph = draw(est_pag_no_fnodes, direction="LR")
dot_graph.render(outfile="psi_pag.png", view=True, cleanup=True)
