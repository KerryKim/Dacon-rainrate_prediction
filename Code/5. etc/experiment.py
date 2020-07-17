import pandas as pd
ensemble = 0
nfolds=4
# Ensemble test_results
for fold in range(nfolds):
    test_result = pd.read_csv('result/baseline_master_200514_{}.csv'.format(fold))
    test_result = pd.DataFrame(test_result).values
    ensemble += test_result[:, 1:] * 1. / nfolds

submission = pd.read_csv('sample_submission.csv')
submission.iloc[:, 1:] = ensemble.reshape(-1, 1600)
submission.to_csv('result/rainrate_ens.csv', index=False)