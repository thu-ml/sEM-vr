//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_FLAGS_H
#define DSLDA_FLAGS_H

#include "gflags/gflags.h"

DECLARE_double(sch_a);
DECLARE_double(sch_t0);
DECLARE_double(sch_p);
DECLARE_int32(sch_bs);
DECLARE_int32(num_batches);
DECLARE_int32(cv_period);
DECLARE_int32(num_iters);
DECLARE_int32(num_iiters);
DECLARE_int32(num_inf_iters);
DECLARE_int32(batch_after);
DECLARE_int32(adapt_after);
DECLARE_string(theta_file);
DECLARE_string(phi_file);

DECLARE_bool(adapt_step);
DECLARE_bool(adapt_dist);
DECLARE_bool(vr);
DECLARE_bool(decay_step);
DECLARE_bool(prior);
DECLARE_bool(proj);
DECLARE_double(max_step);
DECLARE_double(min_step);

DECLARE_double(min_param);
DECLARE_int32(adapt_length);
DECLARE_int32(max_vocab);
DECLARE_int32(test_lag);

DECLARE_int32(num_particles);

DECLARE_int32(scsg_const);

DECLARE_int32(seed);
DECLARE_string(test_alg);
DECLARE_bool(test_on_train);

DECLARE_int32(ssvi_ws);

#endif //DSLDA_FLAGS_H
