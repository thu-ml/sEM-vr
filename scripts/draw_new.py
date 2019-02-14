import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import re

log_dir = '.'
dataset = [#('nips100_a0.100000_b0.010000', [1, 100], [1000, 1200], [0, 10], [1, 100], [0, 0]),
           #('nytimes100_a1.000000_b0.010000', [1, 100], [3500, 4000], [10, 1000], [0, 20], [0, 0]),
           #('pubmed100_a1_b0.01', [1, 10], [2400, 3000], [0, 10], [0, 0], [1800, 2500])]
            ('wiki100_a0.1_b1', [0, 0], [0, 0], [0, 300], [0, 0], [600, 1000]),
            ('wiki200_a1_b1', [0, 0], [0, 0], [0, 1000], [0, 0], [400, 800])]
algs = [('sem', 'sEM', 'b--'),
        ('cvsem', 'cv-sEM', 'r--'),
        ('goem',  'G-OEM', 'k-'),
        ('cvboem', 'CVB-OEM',  'c:'),
        ('icvboem', 'ICVB-OEM', 'b:'),
        ('cvicvboem', 'ICVB-cvOEM', 'r:'),
        ('ncvboem', 'NCVB-OEM', 'b-'),
        ('cvncvboem', 'NCVB-cvOEM', 'r-')]

for d, ixlim, ylim, txlim, test_ixlim, test_ylim in dataset:
    print(d)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    max_time = 0
    for alg, algname, sty in algs:
        # parse batch log likelihood, training and testing perplexity, and time
        f_name = '{}/{}_{}.log'.format(log_dir, d, alg)

        lls = []
        train_perp = []
        test_perp  = []
        train_iters = []
        test_iters = []
        train_times = []
        test_times = []
        train_cnt = 0
        test_cnt = 0
        sub_tests = False
        with open(f_name) as f:
            for line in f.readlines():
                if line.find('Testing perplexity') != -1:
                    sub_tests = True
                    ls = line.split()
                    test_cnt += 0.1
                    test_ppl = float(ls[2])
                    time = float(ls[-1])
                    test_iters.append(test_cnt)
                    test_times.append(time)
                    test_perp.append(test_ppl)

                if line.find('Iteration') != -1:
                    ls = line.split()
                    ls[-1] = re.sub('[^0-9\.]', '', ls[-1])

                    if alg == 'sem' or alg == 'cvsem':
                        train_ppl = float(ls[9])
                        # train = float(l[10])
                        test_ppl = float(ls[13])
                    elif alg == 'cvboem':
                        train_ppl = float(ls[8])
                        test_ppl = float(ls[11])
                    else:
                        train_ppl = float(ls[8])
                        # train = float(l[9])
                        test_ppl = float(ls[12])
                    time = float(ls[-1])

                    train_cnt += 1
                    train_iters.append(train_cnt)
                    train_perp.append(train_ppl)
                    train_times.append(time)
                    if not sub_tests:
                        test_iters.append(train_cnt)
                        test_times.append(time)
                        test_perp.append(test_ppl)

        ax1.plot(train_iters, train_perp, sty, label=algname)
        ax1.legend()
        # ax1.set_xlim(ixlim)
        # if ylim[0] != 0:
        #     ax1.set_ylim(ylim)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training p(w | theta, phi)')

        ax2.plot(train_times, train_perp, sty, label=algname)
        # ax2.set_xlim(txlim)
        ax2.legend()
        # ax2.set_xlim(txlim)

        ax3.plot(test_iters, test_perp, sty, label=algname)
        if test_ixlim[1] != 0:
            ax3.set_xlim(test_ixlim)
        if test_ylim[1] != 0:
            ax3.set_ylim(test_ylim)
        ax3.legend()

        ax4.plot(test_times, test_perp, sty, label=algname)
        ax4.legend()
        if txlim[1] != 0:
            ax4.set_xlim(txlim)
        if test_ylim[1] != 0:
            ax4.set_ylim(test_ylim)
        # ax4.set_xlim(txlim)


    fig1.savefig('{}_iter_ppl.pdf'.format(d))
    fig2.savefig('{}_time_ppl.pdf'.format(d))
    fig3.savefig('{}_iter_tppl.pdf'.format(d))
    fig4.savefig('{}_time_tppl.pdf'.format(d))

    plt.close('all')

print('Complete')
