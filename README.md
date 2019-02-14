# sEM-vr
====

Code for the paper [Stochastic Expectation Maximization with Variance Reduction](http://papers.nips.cc/paper/8021-stochastic-expectation-maximization-with-variance-reduction).

Prerequisites
====

* CMake>=3.2
* C++ compiler that supports C++1y (tested with gcc-5.4)
* OpenMP

Get the repository
====

```
git clone --recursive https://github.com/thu-ml/sEM-vr.git
```


Prepare data
====

You can download the example NIPS dataset, and put it under the `data` folder.

```
mkdir data
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nips.txt
gunzip docword.nips.txt.gz
python ../scripts/convert_data.py --data nips --ntest 50
cd ..
```

Data format
----

The data contains three files, `nips.vocab`, `nips.libsvm.train` and `nips.libsvm.test`. 
The file `nips.vocab` is the vocabulary, where each line is a word in the vocabulary. 
The other two files are the training and testing data set (testing data set is only useful for LDA), which are in libsvm sparse vector format. 

Each line of `nips.libsvm.train` is a document, 

```
N v1:c1 v2:c2 ...
```

where `N` is the number of key-value pairs in this line. Each key-value pair `v1:c1` means that the `v1`-th word in the vocabulary (index starts from 0) occurs `c1` times in this document.

*Note: if you update the `.libsvm` files, please remove the `.d` and `.w` files in the same folders. They are generated automatically by the trainer.*

Build
====

```
./build.sh
cd release
make -j
```

Run
====

Run stochastic EM
```
lda/lda --k 100 --num_iters 20 --alpha_sum 0.1 --beta 0.01 --alg cvsem --novr
```

Run stochastic EM with variance reduction
```
lda/lda --k 100 --num_iters 20 --alpha_sum 0.1 --beta 0.01 --alg cvsem --max_step 0.2
```

The output looks like this
```
{"iter":0, "subiter":-1, "running_time":1.52959, "step_size":0.0816497, "test_ll":-473912, "test_ppl":1846.38, "theta_prior":-1514.84, "phi_prior":-154222, "log_likelihood":-1.34593e+07, "log_posterior":-1.3615e+07, "joint_ppl":1339.37, "testing_time":4.7116}
```

pLSA metrics
* `log_posterior` is the training objective term, which is the summation of 
  - `log_likelihood = log p(words | theta, phi)` is the likelihood term
  - `theta_prior = log p(theta)` is the prior of theta
  - `phi_prior = log p(phi)` is the prior of phi

LDA metrics
* `test_ll = log p(words | phi)` is the likelihood on the test set, which is computed with the left-to-right sampler.
* `test_ppl` is the perplexity on the test set.


License
====

MIT

Citation
====

Please cite our paper if you find this code useful. 

```
@inproceedings{chen2018stochastic,
  title={Stochastic Expectation Maximization with Variance Reduction},
  author={Chen, Jianfei and Zhu, Jun and Teh, Yee Whye and Zhang, Tong},
  booktitle={Advances in Neural Information Processing Systems},
  pages={7978--7988},
  year={2018}
}
```
