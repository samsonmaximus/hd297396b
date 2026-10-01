# Fig. 5 of v14: posterior of the adopted model (Matern-3/2 GP, joint RV + dLW, one circular
# planet, 104 epochs), from the nested-sampling run  python gpns.py 1p in 7  ->  ns_1p_in_s7.pkl
import pickle, numpy as np, corner, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dynesty.utils import resample_equal
r = pickle.load(open('ns_1p_in_s7.pkl', 'rb'))
w = np.exp(r['logwt'] - r['logz']); w /= w.sum()
np.random.seed(3); s = resample_equal(r['samples'], w)
P, K, ph = s[:, 21], s[:, 22], s[:, 23]
# model: -K sin(2 pi (t/P + ph)) ... gpns uses  y - K sin(2 pi (tr/P + ph)),  tr = t - 2455000
# conjunction (RV decreasing through zero for a +K sin model sign convention used in Table 4):
# sin argument = 0 at tr = -ph P  -> T0 = 2455000 - ph P; shift by whole periods near 2456298.
T0 = 2455000.0 - ph * P - P / 2          # heldout_new.py convention: v = -K sin(2 pi (t - Tc)/P)
n = np.round((2456298.34 - T0) / P)
Tc = T0 + n * P - 2450000
print('Tc median', np.median(Tc), '(Table 4: 6298.34 +- 0.12)')
A_rv, ell, j072 = s[:, 9], s[:, 20], s[:, 4]
X = np.column_stack([(P - 4.26837) * 1e4, K, Tc - 6298.34, A_rv, np.log10(ell)])
lab = [r'$\Delta P$ ($10^{-4}$ d)', r'$K$ (m s$^{-1}$)', r'$\Delta T_{\rm conj}$ (d)', r'$A_{\rm RV}$ (m s$^{-1}$)',
       r'$\log_{10}\ell$ (d)']
for i, l in enumerate(lab):
    q = np.percentile(X[:, i], [16, 50, 84]); print(l, q[1], q[2] - q[1], q[1] - q[0])
fig = corner.corner(X, labels=lab, levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)), quantiles=[0.16, 0.5, 0.84],
                    show_titles=False, label_kwargs=dict(fontsize=7), max_n_ticks=3, color='#1f4e79',
                    plot_datapoints=False, range=[0.997]*5, labelpad=0.12, fill_contours=True, smooth=1.0, title_fmt='.3f')
for ax in fig.get_axes():
    ax.tick_params(labelsize=5.5)
fig.set_size_inches(3.6, 3.6); fig.subplots_adjust(wspace=0.06, hspace=0.06)
fig.savefig('fig_corner.pdf', bbox_inches='tight'); fig.savefig('fig_corner.png', dpi=150, bbox_inches='tight')
print('lnZ', r['logz'], 'n', len(s))
