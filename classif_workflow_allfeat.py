import pandas as pd
import sklearn
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (StratifiedKFold, LeaveOneOut,
                                     cross_validate,
                                     permutation_test_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
import matplotlib
import matplotlib.pyplot as plt
%matplotlib

df = pd.read_csv('data/data_thres_0.2.csv')

labels = ["DS" if i == 1 else "CTRL" for i in df['Group']]

X = df.iloc[:, 7:737].values

# read labels: NC & LLD
targets = df['Group'].values

# AdaBoost classifier

pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', AdaBoostClassifier(random_state=0))
])

results = {}
results['fold'] = []
results['balanced_accuracy'] = []
results['accuracy'] = []
results['recall'] = []
results['precision'] = []
results['f1'] = []
results['roc_auc'] = []
results['feat_vals'] = []

scorers = ['balanced_accuracy', 'accuracy', 'recall', 'precision',
           'f1', 'roc_auc']

for i in range(0, 30):
    kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=i)
    # read columns of interest

    scores_out = cross_validate(pipe, X, targets,
                                scoring=scorers,
                                cv=kfold, return_estimator=True,
                                n_jobs=-1)
    print(scores_out['test_balanced_accuracy'].mean())
    results['fold'].append(i)
    results['balanced_accuracy'].append(scores_out['test_balanced_accuracy'].mean())
    results['accuracy'].append(scores_out['test_accuracy'].mean())
    results['recall'].append(scores_out['test_recall'].mean())
    results['precision'].append(scores_out['test_precision'].mean())
    results['f1'].append(scores_out['test_f1'].mean())
    results['roc_auc'].append(scores_out['test_roc_auc'].mean())
    results['feat_vals'].append(scores_out['estimator'][0].named_steps["clf"].feature_importances_)
df_res = pd.DataFrame(results)
df_res.to_csv('results/all_crossVal_30iter.csv')

all_rois = df.iloc[:, 7:737].columns
all_feat = df_res.feat_vals.mean()
ind = all_feat.argsort()[-100:][::-1]
plt.bar(all_rois[ind], all_feat[ind])
plt.xticks(rotation=90)
plt.tight_layout()
plt.rcParams["figure.figsize"] = (12,7.5)
plt.show()
plt.savefig('results/all_feats/results_feat.svg')
plt.close()

plt.figure()
sns.violinplot(data = df_res[scorers])
plt.savefig('results/all_feats/allfeat_violin.svg')


perm_res = {}
perm_res['score'] = []
perm_res['perm_scores'] = []
perm_res['pvalue'] = []
perm_res['scorer'] = []


kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=0)
# perform test
for scorer in ["roc_auc", "balanced_accuracy"]:
    model_score, perm_scores, pvalue = permutation_test_score(
        pipe, X, targets, scoring=scorer, cv=kfold, n_permutations=1000,
        n_jobs=-1)
    perm_res['score'].append(model_score)
    perm_res['perm_scores'].append(perm_scores)
    perm_res['pvalue'].append(pvalue)
    perm_res['scorer'].append(scorer)
    df_perm = pd.DataFrame(perm_res)
    df_perm.to_csv('results/all_feats/res_perm' + scorer + '.csv')
    fig, ax = plt.subplots()

    ax.hist(perm_scores, bins=40, facecolor='coral', density=True)
    ax.axvline(model_score, ls="--", color="r")
    score_label = f"Score on original\ndata: {model_score:.2f}\n(p-value: {pvalue:.3f})"
    print(score_label)
    #ax.text(left, top, score_label, fontsize=12)
    ax.set_xlabel(scorer)
    _ = ax.set_ylabel("Frequency")
    plt.savefig('results/perm_' + scorer + '_hist.svg')
    plt.show()
