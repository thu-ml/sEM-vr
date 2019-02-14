import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import re

log_dir = '../20180430_more_algorithms/'
dataset = [('nips10', [1, 100], [1700, 1900], [0, 10], [0, 0], [0, 0]),
           ('nips100', [1, 100], [1000, 1200], [0, 10], [1, 100], [1500, 1700]),
           ('nytimes10', [1, 20], [6000, 7000], [10, 100], [0, 0], [0, 0]),
           ('nytimes100', [1, 100], [3500, 4000], [10, 1000], [0, 20], [4000, 5000]),
           ('pubmed10', [1, 10], [5500, 6500], [0, 10], [0, 0], [0, 0]),
           ('pubmed100', [1, 10], [2400, 3000], [0, 10], [0, 0], [0, 0])]
algs = [('all_batch', 'bEM', '--'),
        ('a1t10000', 'sEM rho = 1/(t+10000)^0.5', 'r:'),
        ('a1t1000',  'sEM rho = 1/(t+1000)^0.5', 'g:'),
        ('a1t100', 'sEM rho = 1/(t+100)^0.5',  'b:'),
        ('a3t1000', 'sEM rho = 3/(t+1000)^0.5', 'c:'),
        ('a10t10000', 'sEM rho = 10/(t+10000)^0.5', 'm:'),
        ('s0.01', 'cv-sEM rho=0.01', 'r'),
        ('s0.05', 'cv-sEM rho=0.05', 'g'),
        ('s0.1', 'cv-sEM rho=0.1', 'b')]

for d, ixlim, ylim, txlim, test_ixlim, test_ylim in dataset:
    print(d)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    max_time = 0
    for alg, algname, sty in algs:
        # parse batch log likelihood, training and testing perplexity, and time
        f_name = '{}/{}_{}.log'.format(log_dir, d, alg)

        lls = []
        train_perp = []
        test_perp  = []
        times = []
        test_perp_idx = []
        with open(f_name) as f:
            cnt = 0
            total_time = 0
            for line in f.readlines():
                if line.find('Log likelihood') != -1:
                    lls.append(float(line.split()[-1]))
                if line.find('perplexity') != -1:
                    ls = line.split()
                    ls[-1] = re.sub('[^0-9\.]', '', ls[-1])
                    train_perp.append(float(ls[9]))
                    tp = float(ls[12])
                    if tp != 0:
                        test_perp.append(tp)
                        test_perp_idx.append(cnt)

                    t1 = float(ls[-1])
                    if ls[-2] != 'Times':
                        t2 = float(ls[-2])
                    else:
                        t2 = 0
                    total_time += t1 + t2
                    times.append(total_time)
                    cnt += 1

        max_time = max(max_time, total_time)

        xiters = range(1, len(train_perp)+1)
        ax1.plot(xiters, train_perp, sty, label=algname)
        ax1.legend()
        ax1.set_xlim(ixlim)
        if ylim[0] != 0:
            ax1.set_ylim(ylim)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Perplexity')

        ax2.plot(times, train_perp, sty, label=algname)
        ax2.set_xlim(txlim)
        ax2.legend()
        # ax2.set_xlim(txlim)

        ax3.semilogy(np.max(lls) - lls, sty, label=algname)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Suboptimality')
        # ax3.set_xlim(ixlim)
        ax3.legend()

        times = np.array(times)
        test_perp_idx = np.array(test_perp_idx)
        ax4.plot(test_perp_idx+1, test_perp, sty, label=algname)
        if test_ixlim[1] != 0:
            ax4.set_xlim(test_ixlim)
        if test_ylim[0] != 0:
            ax4.set_ylim(test_ylim)
        ax4.legend()

        ax5.plot(times[test_perp_idx], test_perp, sty, label=algname)
        ax5.legend()
        ax5.set_xlim(txlim)


    fig1.savefig('{}_iter_ppl.pdf'.format(d))
    fig2.savefig('{}_time_ppl.pdf'.format(d))
    fig3.savefig('{}_iter_ll.pdf'.format(d))
    fig4.savefig('{}_iter_tppl.pdf'.format(d))
    fig5.savefig('{}_time_tppl.pdf'.format(d))

    plt.close('all')

print('Complete')
