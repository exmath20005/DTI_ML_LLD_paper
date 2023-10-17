import pandas as pd
import sklearn
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, LeaveOneOut,
                                     cross_validate,
                                     permutation_test_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
import matplotlib
import matplotlib.pyplot as plt
%matplotlib

sklearn.metrics.get_scorer_names()

def cm_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}

# df = pd.read_csv('data/data.csv')
df = pd.read_csv('data/data_thres_0.2.csv')

labels = ["DS" if i == 1 else "CTRL" for i in df['Group']]

features = df.iloc[:, 7:737].values
key_words = ['_vol', '_FA', '_trace', '_AD', '_RD']

# read labels: NC & LLD
targets = df['Group'].values
# initiate cross validation method

# set pipelines to be tested
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', AdaBoostClassifier(random_state=0))
])

# Gradient Boosting Classifier
# pipe = Pipeline([
#     (),
#     ('scaler', StandardScaler()),
#     ('clf', GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=1, random_state=0))
# ])

# For svc with feature selection
# need to comment out all features importnce processes
# as SVC return no feature importances

# pipe = Pipeline([
#             ('reduce_dim', SelectKBest(f_classif, k=60)),
#             ('scaler', StandardScaler()),
#             ('clf', SVC(class_weight='balanced'))
#   ])

results = {}
results['fold'] = []
results['metric'] = []
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
    for metric in key_words:
        # read columns of interest
        X = df[[i for i in df.columns if metric in i]].values
        scores_out = cross_validate(pipe, X, targets,
                                    scoring=scorers,
                                    cv=kfold, return_estimator=True,
                                    n_jobs=-1)
    # TP = []
    # FP = []
    # TN = []
    # FN = []
    # for fold in range(0,12):
    #     A = [[1, 1, 1, 1],
    #         [1-scores_out['test_precision'][fold], -scores_out['test_precision'][fold] ,0 ,0],
    #         [1-scores_out['test_recall'][fold], 0, 0, -scores_out['test_recall'][fold]],
    #         [scores_out['test_accuracy'][fold]-1, scores_out['test_accuracy'][fold],
    #          scores_out['test_accuracy'][fold]-1, scores_out['test_accuracy'][fold]]
    #         ]
    #     B = [38, 0, 0, 0]
    #     TP.append(np.linalg.solve(A,B)[0])
    #     FP.append(np.linalg.solve(A,B)[1])
    #     TN.append(np.linalg.solve(A,B)[2])
    #     FN.append(np.linalg.solve(A,B)[3])
        print(metric, scores_out['test_balanced_accuracy'].mean())
        results['fold'].append(i)
        results['metric'].append(metric)
        results['balanced_accuracy'].append(scores_out['test_balanced_accuracy'].mean())
        results['accuracy'].append(scores_out['test_accuracy'].mean())
        results['recall'].append(scores_out['test_recall'].mean())
        results['precision'].append(scores_out['test_precision'].mean())
        results['f1'].append(scores_out['test_f1'].mean())
        results['roc_auc'].append(scores_out['test_roc_auc'].mean())
        results['feat_vals'].append(scores_out['estimator'][0].named_steps["clf"].feature_importances_)
df_res = pd.DataFrame(results)
# df_res.to_csv('results/crossVal_30iter.csv')
df_res[df_res.metric == '_vol'].accuracy.mean()
df_res[df_res.metric == '_trace'].accuracy.mean()
df_res[df_res.metric == '_RD'].accuracy.mean()
df_res[df_res.metric == '_AD'].accuracy.mean()
df_res[df_res.metric == '_FA'].accuracy.mean()

rd_rois = df[[i for i in df.columns if '_RD' in i]].columns
rd_feat = df_res[df_res.metric == '_RD'].feat_vals.mean()
ind = rd_feat.argsort()[-20:][::-1]
plt.bar(rd_rois[ind], rd_feat[ind])
plt.xticks(rotation=90)
plt.tight_layout()
plt.rcParams["figure.figsize"] = (12,7.5)
plt.show()
plt.savefig('results/feat_RD.svg')
plt.close()

trace_rois = df[[i for i in df.columns if '_trace' in i]].columns
trace_feat = df_res[df_res.metric == '_trace'].feat_vals.mean()
ind = trace_feat.argsort()[-20:][::-1]
plt.bar(trace_rois[ind], trace_feat[ind])
plt.xticks(rotation=90)
plt.tight_layout()
plt.rcParams["figure.figsize"] = (12,7.5)
plt.show()
plt.savefig('results/feat_trace.svg')


for scorer in scorers:
    plt.figure()
    sns.violinplot(x ="metric",
                   y =scorer,
                   data = df_res)
    plt.savefig('results/violin_' + scorer + '.svg')



perm_res = {}
perm_res['score'] = []
perm_res['perm_scores'] = []
perm_res['pvalue'] = []
perm_res['metric'] = []

loo = LeaveOneOut()
for metric in key_words:
    X = df[[i for i in df.columns if metric in i]].values
    # replicate pipeline that produces this result

    # perform test
    model_score, perm_scores, pvalue = permutation_test_score(
        pipe, X, targets, scoring="balanced_accuracy", cv=loo, n_permutations=1000,
        n_jobs=-1)
    perm_res['score'].append(model_score)
    perm_res['perm_scores'].append(perm_scores)
    perm_res['pvalue'].append(pvalue)
    perm_res['metric'].append(metric)
df_perm = pd.DataFrame(perm_res)
