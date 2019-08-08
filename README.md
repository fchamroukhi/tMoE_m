>>t mixture of experts (TMoE): Robust mixtures-of-experts modeling using the t distribution <<
 
TMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
heterogeneous data with the t Mixture-of-Experts (TMoE) model.
 
TMoE provides a flexible and robust modeling framework for heterogenous data with possibly
heavy-tailed distributions and corrupted by atypical observations. TMoE consists of a mixture of K
t expert regressors network (of degree p) gated by a softmax gating network (with regression
degree q) and is represented by - The gating net. parameters $\alpha$'s of the softmax net. - The
experts network parameters: The location parameters (regression coefficients) $\beta$'s, scale
parameters $\sigma$'s, and the degree of freedom (robustness) parameters $\nu$'s. TMoE thus
generalises  mixtures of (normal, t, and) distributions and mixtures of regressions with these
distributions. For example, when $q=0$, we retrieve mixtures of (t-, or normal) regressions, and
when both $p=0$ and $q=0$, it is a mixture of (t-, or normal) distributions. It also reduces to
the standard (normal, t) distribution when we only use a single expert (K=1).
 
Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
illustrate the use of the model in model-based clustering of heterogeneous regression data and in
fitting non-linear regression functions. Real-world data examples of tone perception for musical
data analysis, and the one of temperature anomalies for the analysis of climate change data, are
also provided as application of the model.
 
To run it on the provided examples, please run "main_demo_TMoE_SimulatedData.m" or
"main_demo_TMoE_RealData.m"

Please cite the code and the following papers when using this code:
- F. Chamroukhi. Robust mixture of experts modeling using the $t$-distribution. Neural Networks, V. 79, p:20?36, 2016
- F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015

(c) Introduced and written by Faicel Chamroukhi (may 2015)