import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.utils import permutationTest
import pandas as pd
from scipy.stats import ttest_ind


data_dir = "/Volumes/Porter's Data/penn-state/data-sets/reco/results"
model = "ncf"
dataset = "amazon"
task = "choice"
n_perm = 1000
control = "pairwise"
treatment = "pairwise_utility"

if task == "choice":
    prediction = "logit"
elif task == "rating":
    prediction = "mse"
else:
    raise NotImplementedError(task)

control_fname = pd.read_csv("{}/{}/{}_{}_{}_{}.txt".format(data_dir, model, model, dataset, task, control))
treatment_fname = pd.read_csv("{}/{}/{}_{}_{}_{}.txt".format(data_dir, model, model, dataset, task, treatment))

for col in control_fname.columns:
    print("Metric: {}".format(col))
    x_control = control_fname[col].values
    x_treatment = treatment_fname[col].values

    mean_control = x_control.mean()
    mean_treatment = x_treatment.mean()

    print("Mean Control: {:.3f}".format(mean_control))
    print("Mean Treatment: {:.3f}".format(mean_treatment))

    mean_diff = mean_treatment - mean_control

    print("Difference in means: {:.3f}".format(mean_diff))
    p = permutationTest(x_control, x_treatment, n_perm)
    print("Permutation p-val: {:.3f}".format(p))

    t, p_val  = ttest_ind(x_control, x_treatment)
    print("Fisher p-val: {}".format(p_val))