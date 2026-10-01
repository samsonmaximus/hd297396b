import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
 'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'lines.linewidth':0.9,
 'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':True,'figure.dpi':150,'mathtext.fontset':'dejavuserif'})
COL={'pre_072':'#2a78d6','pre_183':'#eb6834','pre_oth':'#1baf7a','post':'#eda100'}
MRK={'pre_072':'o','pre_183':'s','pre_oth':'^','post':'D'}
NAME={'pre_072':'072.C-0488 (GTO)','pre_183':'183.C-0972','pre_oth':'other, pre-2015','post':'post-2015'}
INK='#222222'; MUTED='#8a8a85'; ACC='#c0392b'
W1=3.46; W2=7.1
