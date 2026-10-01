"""Cross-check our hand-rolled GP log-likelihood against george, single series."""
import numpy as np, p3data, p3model, george
from george import kernels

d = p3data.build('serval', clip=4.0)
rng = np.random.default_rng(3)
t = d['t']; y = rng.normal(0, 3, len(t)); e = d['erv']
l, A = 300.0, 4.5

# ours
S = A**2 * p3model.k_m32(np.abs(t[:,None]-t[None,:]), l) + np.diag(e**2)
ours = p3model._mvn(y, S)

# george: Matern32Kernel takes metric = l^2/3  because george uses
# (1+sqrt(3 r2/metric)) exp(-sqrt(3 r2/metric)) with r2=(dt)^2
gp = george.GP(A**2 * kernels.Matern32Kernel(l**2))
gp.compute(t, e)
theirs = gp.log_likelihood(y)
print(f"ours   = {ours:.8f}\ngeorge = {theirs:.8f}\ndiff   = {ours-theirs:.2e}")

# QP cross-check: george ExpSquared * ExpSine2
eta2, eta3, eta4 = 800.0, 31.0, 0.6
Sq = A**2 * p3model.k_qp(np.abs(t[:,None]-t[None,:]), eta2, eta3, eta4) + np.diag(e**2)
ours_q = p3model._mvn(y, Sq)
k = A**2 * kernels.ExpSquaredKernel(eta2**2) * kernels.ExpSine2Kernel(gamma=0.5/eta4**2, log_period=np.log(eta3))
gp2 = george.GP(k); gp2.compute(t, e)
print(f"QP ours   = {ours_q:.8f}\nQP george = {gp2.log_likelihood(y):.8f}\ndiff = {ours_q-gp2.log_likelihood(y):.2e}")

# Student-t sanity: nu -> large must approach the Gaussian
for nu in (1e3, 1e5, 1e7):
    print(f"  nu={nu:9.0e}  MVT-MVN = {p3model._mvn(y,S,nu)-ours:+.4f}")
