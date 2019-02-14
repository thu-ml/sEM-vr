//
// Created by jchen on 19/03/18.
//

void main() {
    // Sample delta_theta and delta_phi
    if (iter == -1) {
        double first_var = 0;
        double cross_var = 0;
        double second_var = 0;
        double first_t = 0;
        double real_var = 0;
        double theta_var = 0;
        double real_theta_var = 0;
        double phi_var = 0;
        //int    eval_size = corpus.T;
        int eval_size = 10000;
        int eval_iters = 100;
        // New sampling scheme
        for (int iter = 0; iter < eval_iters; iter++) {
            theta = theta_0;
            phi = phi_0;

            fill(cdk.begin(), cdk.end(), 0);
            fill(cvk.begin(), cvk.end(), 0);
            fill(ck.begin(), ck.end(), 0);
            for (int d = 0; d < corpus.D; d++) {
                int D = corpus.w[d].size();
                if (!D) continue;
                int M = max(1, D / multi);
                double s = (double) D / M;
                shuffle(corpus.w[d].begin(), corpus.w[d].end(), generator);
                for (int i = 0; i < M; i++) {
                    int v = corpus.w[d][i];
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = theta[d * K + k] * phi[v * K + k];
                    sum = s / sum;
                    for (int k = 0; k < K; k++)
                        cdk[d * K + k] += prob[k] * sum;
                }
            }
            for (int v = 0; v < corpus.V; v++) {
                int V = corpus.d[v].size();
                if (!V) continue;
                int M = max(1, V / multi);
                double s = (double) V / M;
                shuffle(corpus.d[v].begin(), corpus.d[v].end(), generator);
                for (int i = 0; i < M; i++) {
                    int d = corpus.d[v][i];
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = theta[d * K + k] * phi[v * K + k];
                    sum = s / sum;
                    for (int k = 0; k < K; k++) {
                        cvk[v * K + k] += prob[k] * sum;
                        ck[k] += prob[k] * sum;
                    }
                }
            }
            ThetaPhi();

            shuffle(corpus.tokens.begin(), corpus.tokens.end(), generator);
            for (int i = 0; i < eval_size; i++) {
                auto &token = corpus.tokens[i];

                auto *h_t = theta.data() + token.d * K;
                auto *h_p = phi.data() + token.v * K;
                auto *b_t = theta_bar.data() + token.d * K;
                auto *b_p = phi_bar.data() + token.v * K;
                auto *t_t = theta_0.data() + token.d * K;
                auto *t_p = phi_0.data() + token.v * K;
                auto *p_t = theta_p.data() + token.d * K;
                auto *p_p = phi_p.data() + token.v * K;

                double dtdp = 0;
                double ptdp = 0;
                double dtpp = 0;
                double p = 0;
                double l = 0;
                for (int k = 0; k < K; k++) {
                    dtdp += (h_t[k] - b_t[k]) * (h_p[k] - b_p[k]);
                    ptdp += p_t[k] * (h_p[k] - b_p[k]);
                    dtpp += (h_t[k] - b_t[k]) * p_p[k];
                    p += p_t[k] * p_p[k];
                    l += ((1 - step_size) * t_t[k] + step_size * h_t[k]) *
                         ((1 - step_size) * t_p[k] + step_size * h_p[k]);
                }

                first_var += -0.5 * dtpp * dtpp / (p * p);
                cross_var += -ptdp * dtpp / (p * p);
                second_var += -0.5 * ptdp * ptdp / (p * p);
                first_t += dtdp / p;
                real_var += log(l) - log(p);
            }

            for (int d = 0; d < corpus.D; d++)
                for (int k = 0; k < K; k++) {
                    double t     = theta[d*K+k];
                    double tbar  = theta_bar[d*K+k];
                    double tp    = theta_p[d*K+k];
                    double t0    = theta_0[d*K+k];
                    double delta = t - tbar;
//                    theta_var += alpha * step_size * delta / tp - 0.5 * alpha * step_size * step_size * delta * delta / (tp * tp);
                    theta_var += -0.5 * alpha * step_size * step_size * delta * delta / (tp * tp);
                    real_theta_var += alpha * (log(t0*(1-step_size) + t*step_size) - log(tp));
                }

            for (int v = 0; v < corpus.V; v++)
                for (int k = 0; k < K; k++) {
                    double p     = phi[v*K+k];
                    double pbar  = phi_bar[v*K+k];
                    double pp    = phi_p[v*K+k];
                    double delta = p - pbar;
                    phi_var += -0.5 * beta * step_size * step_size * delta * delta / (pp * pp);
                }
        }

        first_var *= (double) corpus.T / eval_size / eval_iters;
        cross_var *= (double) corpus.T / eval_size / eval_iters;
        second_var *= (double) corpus.T / eval_size / eval_iters;
        first_t *= (double) corpus.T / eval_size / eval_iters;
        real_var *= (double) corpus.T / eval_size / eval_iters;
        theta_var /= eval_iters;
        real_theta_var /= eval_iters;
        phi_var /= eval_iters;
        var = step_size * step_size * (first_var + cross_var + second_var + first_t);
        cout << "Variances " << first_var << ' ' << cross_var << ' ' << second_var << ' '
             << first_t << ' ' << var << ' ' << real_var << endl;
        cout << "Prior variances " << theta_var << ' ' << phi_var << endl;
    }


    if (iter == -1) {
        int D = corpus.w[0].size();
        int M = max(1, D/multi);
        double s = (double)D / M;

        int num_iters = 100000;
        double var_term = 0;
        double taylor_first = 0;
        double taylor_var = 0;
        double theta_bar2 = 0;

        std::vector<double> gammas;
        for (auto v: corpus.w[0]) {
            GetPosterior(theta_0.data(), phi_0.data()+v*K, prob.data());
            gammas.push_back(prob[0]);
        }
        double cdk_whole = 0;
        for (int i = 0; i < D; i++)
            cdk_whole += gammas[i];
        double true_theta_p = (cdk_whole + alpha) / (D + alpha_bar);
        cout << "Theta bar " << theta_bar[0] << ' ' << true_theta_p << endl;
        true_theta_p = (1-step_size)*theta_0[0] + step_size*true_theta_p;
        cout << "Theta p " << true_theta_p << ' ' << theta_p[0] << endl;

        for (int it = 0; it < num_iters; it++) {
            shuffle(gammas.begin(), gammas.end(), generator);
            fill(cdk.begin(), cdk.begin()+K, 0);
            for (int i = 0; i < M; i++)
                cdk[0] += s * gammas[i];
//            for (int i = 0; i < D; i++)
//                cdk[0] += gammas[i];

            theta[0] = (cdk[0] + alpha) / (D + alpha_bar);
            double t = (1-step_size)*theta_0[0] + step_size*theta[0];
            double tp = theta_p[0];
            var_term += log(t) - log(tp);

            double delta_t = theta[0] - theta_bar[0];
//            theta_bar2 += cdk[0];
            theta_bar2 += theta[0];
            taylor_first += step_size * delta_t / tp;
            taylor_var += -0.5 * step_size * step_size * (delta_t * delta_t) / (tp * tp);
        }

//        cout << theta_bar2/num_iters << ' ' << cdk_whole << endl;
        cout << theta_bar2/num_iters << ' ' << theta_bar[0] << endl;
        cout << taylor_first/num_iters << endl;
        cout << "True and Taylor " << var_term/num_iters << ' ' << taylor_var/num_iters << endl;

        double first1 = 0;
        double first2 = 0;
        double R1 = (double)D / M * (M-1) / (D-1) - 1;
        double R2 = (1 - (double)(M-1) / (D-1)) * D / M;
        double Z  = D + alpha_bar;
        double est = 0;
        for (int i = 0; i < D; i++) {
            first1 += gammas[i] / Z;
            first2 += (gammas[i] * gammas[i]) / (Z * Z);
        }
        double tp = theta_p[0];
//        est = -0.5 * step_size * step_size * (R1 * first1 * first1 + R2 * first2) / (tp * tp);
        cout << first1 << ' ' << theta1_bar[0] << " : " << first2 << ' ' << theta2_bar[0] << endl;
        est = -0.5 * step_size * step_size * (R1 * theta1_bar[0] * theta1_bar[0] + R2 * theta2_bar[0]) / (tp * tp);
        cout << "Est " << est << endl;
    }
}