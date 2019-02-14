//
// Created by jianfei on 2018/1/24.
//

#include "flags.h"

DEFINE_double(sch_a, 1.0, "Learning rate scale");
DEFINE_double(sch_t0, 100, "Learning rate t0");
DEFINE_double(sch_p, 0.5, "Learning rate power");
DEFINE_int32(sch_bs, 10000, "Batch size");
DEFINE_int32(num_batches, 50, "Number of minibatches per epoch");
DEFINE_int32(cv_period, 50, "The period to update CV");
DEFINE_int32(num_iters, 100, "Number of iterations");
DEFINE_int32(num_iiters, 100, "Number of inference iterations");
DEFINE_int32(num_inf_iters, 5, "Number of inference iterations");
DEFINE_int32(batch_after, 10000, "Perform batch algorithm instead of stochastic algorithm after how many iterations.");
DEFINE_int32(adapt_after, 10, "Adapting the step size after a certain number of iterations");
DEFINE_string(theta_file, "", "Known theta file");
DEFINE_string(phi_file, "", "Known phi file");

DEFINE_bool(adapt_step, true, "Whether to adapt step size");
DEFINE_bool(adapt_dist, true, "Whether to adapt sampling distribution");
DEFINE_bool(vr, true, "Variance reduction");
DEFINE_bool(decay_step, false, "Decaying step size");
DEFINE_bool(prior, false, "Whether to include prior");
DEFINE_bool(proj, false, "Positive projection");

DEFINE_double(max_step, 1.0, "Maximum step size");
DEFINE_double(min_step, 1e-8, "Minimum step size");

DEFINE_double(min_param, 1e-10, "Min parameter");

DEFINE_int32(adapt_length, 5, "Adapt length");
DEFINE_int32(max_vocab, 10000000, "Max vocabulary");
DEFINE_int32(test_lag, -1, "Number of iiters to test");
DEFINE_int32(num_particles, 10, "Number of particles");

DEFINE_int32(scsg_const, 100, "SCSG constant");
DEFINE_int32(seed, 1, "Random seed");

DEFINE_string(test_alg, "lr", "Testing algorithm, lr or cvb0");
DEFINE_bool(test_on_train, false, "Whether test on training set");

DEFINE_int32(ssvi_ws, 10, "Window size for smoothed svi");
