from obspy.core import read
from palplots.visualize import Plot
import matplotlib.pyplot as plt

stream = read('https://ndownloader.figshare.com/files/6994493','H5',apply_calib=True)

# -----------
# wiggle plot
# -----------
# normal use:
# figure will pop up, but you can't manipulate figure

Plot().wiggle(stream,dimension='theta')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax
# to show figure, use plt.show()

fig,ax = Plot().wiggle(stream,dimension='theta',show=False)
ax.set_ylim((100,0))
plt.show()

# ------------
# contour plot
# ------------
# normal usage:
# figure will pop up, but you can't manipulate figure
Plot().contour(stream,dimension='theta')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax, and cbar
# to show figure, use plt.show()

fig,ax,cbar = Plot().contour(stream,dimension='theta',show=False)
ax.set_ylim((100,0))
cbar.set_clim(-1,1)
plt.show()


# ----------
# fk spectra
# ----------
# normal usage:
# figure will pop up, but you can't manipulate figure

Plot().fk(stream,dimension='theta')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax
# to show figure, use plt.show()

fig, ax, stream_fft, dim = Plot().fk(stream,dimension='theta',show=False)
ax.set_ylim((-2,2))
ax.set_title('F-K Spectra')
plt.show()

# ---------
# fk filter
# ---------
# normal usage:
# figure will pop up, but you can't manipulate figure

Plot().fkfilter(stream,dimension='theta',colormap='gray')

# flexible use:
# figure will not pop up, and you can edit the figure via fig and ax
# to show figure, use plt.show()

fig, ax, filtered_stream, H = Plot().fkfilter(stream,dimension='theta',show=False)
ax.set_ylim((50,0))
ax.set_title('Filtered F-K Spectra')
plt.show()
