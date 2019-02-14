import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import json
sns.set()

# datasets = ['wiki100', 'wiki200'] #['nips100', 'nytimes100', 'pubmed100'], 'wiki100']
datasets = ['nips50', 'nips100', 'nytimes50', 'nytimes100', 'wiki50', 'wiki100', 'pubmed50', 'pubmed100']
settings = ['sem'] #, 'cvsem', 'goem', 'cvboem', 'icvboem', 'cvicvboem', 'ncvboem', 'cvncvboem']
alphas = [0.1, 1, 10, 100]#[0.1, 1, 2, 5, 10, 20, 50, 100]
betas  = [0.01, 0.1, 1]

for d in datasets:
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    params = [0] * len(alphas) * len(betas)
    data1 = np.zeros((len(alphas)*len(betas), len(settings)))
    data2 = np.zeros((len(alphas)*len(betas), len(settings)))
    data3 = np.zeros((len(alphas) * len(betas), len(settings)))

    for i, s in enumerate(settings):
        cnt = 0
        for a in alphas:
            for b in betas:
                log_file = '{}_a{}_b{}_{}.log'.format(d, a, b, s)
                with open(log_file) as f:
                    train_l = 0
                    train = 0
                    test = 0
                    for l in f:
                        if l.find('log_posterior') != -1:
                            res = json.loads(l)
                            train_l = res['log_posterior']
                            train = res['joint_ppl']
                            test = res['test_ppl']

                    print('{} {} a={} b={}\t train={}\ttrain={}\ttest={}'.format(d, s, a, b, train_l, train, test))

                    data1[cnt, i] = train_l
                    data2[cnt, i] = train
                    data3[cnt, i] = test
                
                params[cnt] = 'a{}_b{}'.format(a, b)
                cnt += 1

    sns.heatmap(data1, annot=True, ax=ax1, fmt='.0f', cmap="Blues", robust=True, xticklabels=settings, yticklabels=params)
    sns.heatmap(data2, annot=True, ax=ax2, fmt='.0f', cmap="Blues", robust=True, xticklabels=settings, yticklabels=params)
    sns.heatmap(data3, annot=True, ax=ax3, fmt='.0f', cmap="Blues", robust=True, xticklabels=settings, yticklabels=params)

    fig1.savefig('{}-train-l.pdf'.format(d))
    fig2.savefig('{}-train.pdf'.format(d))
    fig3.savefig('{}-test.pdf'.format(d))

    plt.close('all')
