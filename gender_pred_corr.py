import pandas as pd
import sklearn

import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (StratifiedKFold, LeaveOneOut,
                                     cross_validate,
                                     permutation_test_score)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
import matplotlib
import matplotlib.pyplot as plt
%matplotlib

#sklearn.metrics.get_scorer_names()

# df = pd.read_csv('data/data.csv')
df = pd.read_csv('data/data_thres_0.2.csv')

labels = ["DS" if i == 1 else "CTRL" for i in df['Group']]

features = df.iloc[:, 7:737].values
key_words = ['_vol', '_FA', '_trace', '_AD', '_RD']

# read labels: NC & LLD
targets = df['GENDER'].values

# AdaBoost classifier

pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', AdaBoostClassifier(random_state=0))
])

pred = {}
pred['real'] = []
pred['pred'] = []

for i in range(0, 30):
    kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=i)
    for metric in key_words:
        # read columns of interest
        X = df[[i for i in df.columns if metric in i]].values
        # scores_out = cross_validate(pipe, X, targets,
        #                             scoring=scorers,
        #                             cv=kfold, return_estimator=True,
        #                             n_jobs=-1)
        y_train_pred = cross_val_predict(pipe,X,targets,cv=kfold)
        pred['real'].append(targets)
        pred['pred'].append(y_train_pred)


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
df_res.to_csv('results/gender_crossVal_30iter.csv')
df_res[df_res.metric == '_vol'].accuracy.mean()
df_res[df_res.metric == '_trace'].accuracy.mean()
df_res[df_res.metric == '_RD'].accuracy.mean()
df_res[df_res.metric == '_AD'].accuracy.mean()
df_res[df_res.metric == '_FA'].accuracy.mean()

for scorer in scorers:
    plt.figure()
    sns.violinplot(x ="metric",
                   y =scorer,
                   data = df_res)
    plt.savefig('results/gender_violin_' + scorer + '.svg')


cores={}
cores['r'] = []
cores['p'] = []
for i in range(1,len(pred['pred'])):
    r, p = pearsonr(pred['pred'][i], pred['real'][i])
    cores['r'].append(r)
    cores['p'].append(p)
np.tan(np.arctanh(cores['r']).mean())
fig, ax = plt.subplots()
ax.hist((cores['r']), 60, density=True)
