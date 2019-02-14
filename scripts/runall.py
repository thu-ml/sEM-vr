#!/usr/bin/env python

settings = [('sem', '--alg cvsem --novr'),
            ('cvsem', '--alg cvsem --min_step 0.1 --max_step 0.2'),
            ('goem', '--alg goem --novr'),
            ('cvboem', '--alg cvboem --novr'),
            ('icvboem', '--alg icvboem --novr'),
            ('cvicvboem', '--alg icvboem --max_step 0.1'),
            ('ncvboem', '--alg ncvboem --novr'),
            ('cvncvboem', '--alg ncvboem --max_step 0.1')
]

datasets = [('nips100', 'srun -p small', '--k 100 --num_iters 20'),
            ('nytimes100', 'srun -p medium -c 28', '--prefix ../data/nytimes --k 100 --num_batches 500 --num_iters 20'),
            ('pubmed100', 'srun -p large -c 28', '--prefix ../data/pubmed --k 100 --num_batches 500 --num_iters 5 --test_lag 50'),
            ('pubmed300', 'srun -p large -c 28', '--prefix ../data/pubmed --k 100 --num_batches 500 --num_iters 5 --test_lag 50'),
            ('wiki100', 'srun -p large -c 28', '--prefix ../data/wiki --k 100 --num_batches 500 --num_iters 5 --test_lag 50')]

alphas = [0.1, 1, 10, 100]
betas  = [0.01, 0.1, 1]

for data_p, data_r, data_c in datasets:
    f = open(data_p + '.sh', 'w')
    for a in alphas:
        for b in betas:
            for set_p, set_c in settings:
                log_file = '{}_a{}_b{}_{}.log'.format(data_p, a, b, set_p)
                command = '{} lda/lda {} --alpha_sum {} --beta {} {} '.format(data_r, data_c, a, b, set_c)
                f.write('nohup {} > {} &\n'.format(command, log_file))
    f.close()
