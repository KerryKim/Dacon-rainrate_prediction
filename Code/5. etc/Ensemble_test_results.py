# ensemble test_results
import pandas as pd

nfolds=5
ensemble=0

# ensemble test_results
for fold in range(nfolds):
    test_result = pd.read_csv('result/baseline_master_{}.csv'.format(fold))
    test_result = pd.DataFrame(test_result).values
    ensemble += test_result[:, 1:] * 1. / nfolds

print(ensemble.shape)
submission = pd.read_csv('sample_submission.csv')
submission.iloc[:, 1:] = ensemble.reshape(-1, 1600)
submission.to_csv('result/rainrate_ens.csv', index=False)


'''
a = pd.read_csv('result/baseline_master_0.csv')
a = pd.DataFrame(a).values
a = a[:, 1:]

#print(a)

b = pd.read_csv('result/baseline_master_1.csv')
b = pd.DataFrame(b).values
b = b[:, 1:]
#print(b)

c = (a+b)/2
#print(c)

submission = pd.read_csv('sample_submission.csv')
submission.iloc[:, 1:] = c.reshape(-1, 1600)
submission.to_csv('result/rainrate_ens.csv', index=False)

'''

'''
# Ensemble 3-folds
print('####Ensemble####')
for fold in range(nfolds):
    ensemble += pd.read_csv('result/baseline_master_{}.csv'.format(fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble)
sub_file = 'result/rainrate_ens.csv'
ensemble.to_csv(sub_file, index=False)
'''