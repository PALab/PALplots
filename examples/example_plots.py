from obspy.core import read
from palplots.visualize import contour, wiggle, fk, fkfilter
import matplotlib.pyplot as plt

stream = read('S8-rotd-postreaction.h5','H5',apply_calib=True)

# -----------
# wiggle plot
# -----------
# normal use:
# figure will pop up, but you can't manipulate figure

wiggle(stream,dimension='theta')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax
# to show figure, use plt.show()

fig,ax = wiggle(stream,dimension='theta',show=False)
ax.set_ylim((100,0))
plt.show()

# ------------
# contour plot
# ------------
# normal usage:
# figure will pop up, but you can't manipulate figure
contour(stream,dimension='theta')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax, and cbar
# to show figure, use plt.show()

fig,ax,cbar = contour(stream,dimension='theta',show=False)
ax.set_ylim((100,0))
cbar.set_clim(-1,1)
plt.show()
