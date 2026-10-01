# v15: rotation-period estimate from the mean activity level, on the scale the calibrations were built for.
# Input: Gomes da Silva et al. (2021, A&A 646, A77; VizieR J/A+A/646/A77) for HD 297396:
#   S_MW median 0.4876, log R'HK median -4.787 (weighted mean -4.806 +- 0.004), B-V 1.120 +- 0.003.
# HARPS-RVBank (Perdelwitz et al. 2024) lists log R'HK = -4.64 (median of RHKp), computed with a
# PHOENIX-based photospheric correction; it is shown for comparison only.
import numpy as np
from scipy.optimize import brentq
def noyes(lr, bv):
    y = 5 + lr; logRo = 0.324 - 0.400*y - 0.283*y**2 - 1.325*y**3           # Noyes et al. 1984, eq. (3)
    x = 1 - bv; ltc = 1.362 - 0.166*x + 0.025*x**2 - 5.323*x**3 if x > 0 else 1.362 - 0.14*x   # eq. (4)
    return 10**logRo, 10**ltc
def mh08(lr): return 0.808 - 2.966*(lr + 4.52)                               # Mamajek & Hillenbrand 2008, eq. (5)
# consistency: log R'HK from S_MW with the Middelkoop/Noyes conversion
S, bv = 0.4876, 1.12
lccf = 1.13*bv**3 - 3.91*bv**2 + 2.84*bv - 0.47; rph = 10**(-4.898 + 1.918*bv**2 - 2.893*bv**3)
print("log R'HK from S_MW:", round(np.log10(1.34e-4*10**lccf*S - rph), 3))
for lab, lr, b in [('GdS21 median', -4.787, 1.12), ('GdS21 weighted mean', -4.806, 1.12), ('RVBank (v14 input)', -4.64, 1.09)]:
    Ro, tc = noyes(lr, b)
    print(f"{lab:22s} log R'HK {lr:+.3f}  tau_c {tc:.1f} d  Ro {Ro:.3f}  P_rot(Noyes) {Ro*tc:.1f} d "
          f"[{Ro*tc/10**0.1:.1f}-{Ro*tc*10**0.1:.1f} for 0.1 dex]  P_rot(MH08) {mh08(lr)*tc:.1f} d")
tc = noyes(-4.79, 1.12)[1]; Ro0 = 4.26837/tc
print(f'Rossby number if P_rot = P0: {Ro0:.3f};  implied log R\'HK: Noyes {brentq(lambda l: noyes(l,1.12)[0]-Ro0, -4.6, -3.5):.2f}, '
      f'MH08 {-4.52-(Ro0-0.808)/2.966:.2f}')
print('P0/P_rot for 30-50 d:', round(4.26837/50, 3), '-', round(4.26837/30, 3), '; multiples in 30-50 d:', [k for k in range(1, 15) if 30 <= k*4.26837 <= 50])
print('fractional period stability for 15 deg over 1531 cycles:', round(15/360/1531, 7), '-> for 39 d:', round(15/360/1531*39*1440, 2), 'min')
