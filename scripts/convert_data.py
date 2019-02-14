import argparse
import shutil
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data from uci to libsvm')
    parser.add_argument('--data', type=str, default='nips', help='Dataset to convert. nips, nytimes or pubmed')
    parser.add_argument('--ntest', type=int, default=300, help='Number of testing documents')
    args = parser.parse_args()

    vector_in = 'docword.{}.txt'.format(args.data)
    train_file = '{}.libsvm.train'.format(args.data)
    test_file = '{}.libsvm.test'.format(args.data)

    f_in = open(vector_in)
    train_out = open(train_file, 'w')
    test_out = open(test_file, 'w')

    D = int(f_in.readline())
    V = int(f_in.readline())
    N = int(f_in.readline())
    print('Read {} documents, {} words, {} elements.'.format(D, V, N))
    ntest = min(args.ntest, D)
    ntrain = D - ntest

    perm = np.random.permutation(D)
    test_set = set(perm[:ntest])

    data = [[] for d in range(D)]

    for line in f_in:
        d, v, c = [int(i) for i in line.split()]
        d -= 1
        v -= 1

        data[d].append( (v, c) )

    for d, doc in enumerate(data):
        output = test_out if d in test_set else train_out

        output.write(str(len(doc)))
        for k, v in doc:
            output.write(' {}:{}'.format(k, v))
        output.write('\n')

    f_in.close()
    train_out.close()
    test_out.close()

    shutil.copyfile('vocab.{}.txt'.format(args.data), '{}.vocab'.format(args.data))
