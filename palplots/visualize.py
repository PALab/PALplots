import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from obspy.core import read
from obspy import Trace, Stream
import csv 
import scipy.signal

class Plot(object):
    '''
    Class for plotting stream produced by PLACE automation.  Data must be in H5 format. 
    
    Example usage:
    
         from obspy.core import read
         from obspy import Stream
         from PALplots import wiggle contour, fk, fkfilter, removeDelay
    
         stream = read('PA_2.h5', format='H5', apply_calib=True)
         stream = removeDelay(stream)
         wiggle(stream, percent=50)
         contour(stream)
         fk(stream)
         fkfilter(stream,spread=7,colormap='gray')
    May 27, 2014
    @author: Jami Johnson
    '''   

    def __init__(self):
        self.px = [] #array of picked x-values
        self.py = [] #array of picked y-values
        self.fig, self.ax = plt.subplots()

    def wiggle(self, stream, dimension='x', dx = 1, percent=100, output='points.csv',show=True):
        '''
        Creates plot of traces ("wiggles") vs. x-position from an ObsPy stream produced by PLACE.  
        The positive peaks of each wiggle are filled.
        Parameters: 
             stream : ObsPy stream created with custom header information by PLACE.  Required header information: delta, npts, position, position_unit
             dx : spacing between wiggles.  For example, dx = 10 will plot every tenth trace 
             percent : percent of maximum amplitude for each trace.  A value of 100 plots the full amplitude of the (normalized) traces, a value of 50 clips data with an amplitude greater than 50% of the maximum amplitude in the trace. 
             output : name of file to save 'picked' points to. Saves by default to 'points.csv'
        Returns fig and ax to manipulate plot.
        '''

        newstream = stream.copy() # create copy of stream to manipulate
        percent/=100.

        # ----------
        # Setup plot
        # ----------
        if dimension == 'x':
            newstream.sort(keys=['x_position'])
            inc = newstream[1].stats.x_position - newstream[0].stats.x_position # spacing between x-position
        elif dimension == 'theta':
            newstream.sort(keys=['theta_position'])
            inc = newstream[1].stats.theta_position - newstream[0].stats.theta_position # spacing between angles (theta)

        times = np.arange(0,newstream[0].stats.npts*newstream[0].stats.delta, newstream[0].stats.delta)*1e6

        # ----
        # Plot
        # ----
        for trace in newstream: 

            # Format trace for wiggle plot
            trace.data = np.clip(trace.data,-max(trace.data)*percent,max(trace.data)*percent)# clip data by specified percent of maximum amplitude
            trace.normalize()
            trace.data*=inc
            if dimension == 'x':

                trace.data = trace.data + trace.stats.x_position # arrange to plot vs position
                self.ax.plot(trace.data, times, color='black',picker=True) # plot traces

                # Fill under positive peaks
                if newstream[1].stats.x_position > newstream[0].stats.x_position:
                    self.ax.fill_betweenx(times,trace.data, trace.stats.x_position, where=trace.data>trace.stats.x_position,color='black')
                elif newstream[1].stats.x_position < newstream[0].stats.x_position:
                    self.ax.fill_betweenx(times,trace.data, trace.stats.x_position, where=trace.data<trace.stats.x_position,color='black')

                plt.xlim((newstream[0].stats.x_position-inc,newstream[len(newstream)-1].stats.x_position+inc))
                plt.xlabel('Position ('+str(trace.stats.x_unit)+')')

            elif dimension == 'theta':

                trace.data = trace.data + trace.stats.theta_position # arrange to plot vs position
                self.ax.plot(trace.data, times, color='black',picker=True) # plot traces

                # Fill under positive peaks
                if newstream[1].stats.theta_position > newstream[0].stats.theta_position:
                    self.ax.fill_betweenx(times,trace.data, trace.stats.theta_position, where=trace.data>trace.stats.theta_position,color='black')
                elif newstream[1].stats.theta_position < newstream[0].stats.theta_position:
                    self.ax.fill_betweenx(times,trace.data, trace.stats.theta_position, where=trace.data<trace.stats.theta_position,color='black')

                plt.xlim((newstream[0].stats.theta_position-inc,newstream[len(newstream)-1].stats.theta_position+inc))

                plt.xlabel('Position('+str(trace.stats.theta_unit)+')')

        plt.ylim((trace.stats.npts*trace.stats.delta*1e6,0))
        plt.ylabel('Time ($\mu$s)')

        self.fig.canvas.mpl_connect('button_press_event', self.picker) # select/remove point
        if show == True:
            plt.show()

        # ---------------------------
        # Write picked points to file
        # ---------------------------
        if len(self.px) > 0:
            with open(output,'wb') as csvfile:
                outfile = csv.writer(csvfile, delimiter=',')
                outfile.writerow(self.px)
                outfile.writerow(self.py)

        return self.fig, self.ax

    def contour(self, stream, dimension='x',output='points.csv', colormap='seismic',show=True):
        '''
        Creates 2D image of stream data produced by PLACE.  
        Parameters: 
             stream : ObsPy stream created with custom header information by PLACE.  Required header information: delta, npts, position, position_unit
             colormap : choose colorscheme to display data (e.g. 'jet', 'gray', 'seismic')
             output : name of file to save 'picked' points to. Saves by default to 'points.csv'
        Returns figure, axes, and colorbar.
        '''

        if dimension == 'x':
            stream.sort(keys=['x_position'])
            array = np.rot90(np.array(stream),1)
            plt.imshow(array,extent=[stream[0].stats.x_position,stream[len(stream)-1].stats.x_position,0,stream[0].stats.npts*stream[0].stats.delta*1e6],aspect='auto',cmap=colormap,picker=True)
            plt.xlabel('Scan Location ('+stream[0].stats.x_unit+')')

        elif dimension == 'theta':
            stream.sort(keys=['theta_position'])
            array = np.rot90(np.array(stream),1)
            plt.imshow(array,extent=[stream[0].stats.theta_position,stream[len(stream)-1].stats.theta_position,0,stream[0].stats.npts*stream[0].stats.delta*1e6],aspect='auto',cmap=colormap,picker=True)
            plt.xlabel('Scan Location ('+stream[0].stats.theta_unit+')')

        self.ax.autoscale(False)
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        if stream[0].stats.calib_unit.rstrip() == 'nm/V':
            cbar.set_label('Displacement (nm)')
        elif stream[0].stats.calib_unit.rstrip() == 'mm/s/V':
            cbar.set_label('Particle Velocity (mm/s)')

        plt.ylabel('Time ($\mu$s)')
        self.fig.canvas.mpl_connect('button_press_event', self.picker) # pick point
        if show == True:
            plt.show()

        # ---------------------------
        # Write picked points to file
        # ---------------------------
        if len(self.px) > 0:
            with open(output,'wb') as csvfile:
                outfile = csv.writer(csvfile, delimiter=',')
                outfile.writerow(self.px)
                outfile.writerow(self.py)

        return self.fig, self.ax, cbar

    def fk(self, stream, dimension='x',colormap='gray', output='points.csv',show=True):
        '''
        Plots frequency-wavenumber spectrum for stream recorded by PLACE Scan.py.
        Parameters:
             stream : ObsPy stream created with custom header information defined by Scan.py.  Required header information: delta, npts, position.
             output : filename that selected velocity points are saved to.
        Use left-click to select points.  Each point defines a line with the origin with a slope corresponding to an apparent velocity.  This velocity is displayed when point is chosen.
        To remove a point, right click.
        Returns figure and axes (to manipulate plots), FFT data and spatial dimension used ('x' or 'theta')
        '''

        # ---------------------
        # Setup data/parameters
        # ---------------------
        if dimension == 'x':
            stream.sort(keys=['x_position'])
            dx = stream[1].stats.x_position-stream[0].stats.x_position
        elif dimension == 'theta':
            stream.sort(keys=['theta_position'])
            dx = stream[1].stats.theta_position-stream[0].stats.theta_position
            print 'NOTICE: Spatial frequency here is 1/deg'
        nx = len(stream)
        nt = stream[0].stats.npts
        nk = 2*nx
        nf = 2*nt
        dt = stream[0].stats.delta

        # --------------
        # Compute 2D-FFT
        # --------------
        stream_fft = fft.fftshift(fft.fft2(stream,s=[nk,nf])) #zero padded
        stream_psd2D = np.abs(stream_fft)**2 #2D power spectrum
        stream_psd2D = np.rot90(stream_psd2D,1)

        # ----
        # Plot
        # ----
        plt.imshow(np.log10(stream_psd2D),extent=[-1/(2*dx),1/(2*dx),-1e-6/(2*dt),1e-6/(2*dt)],aspect='auto',cmap = colormap,picker=True)
        self.ax.autoscale(False)
        plt.ylim((0,1e-6/(2*dt)))
        plt.ylabel('Frequency (MHz)')
        if dimension == 'x':
            plt.xlabel('Spatial Frequency (1/'+str(stream[0].stats.x_unit)+')')
        if dimension == 'theta':
            plt.xlabel('Spatial Frequency (1/'+str(stream[0].stats.theta_unit)+')')
        self.fig.canvas.mpl_connect('button_press_event', self.pickV) # pick/remove points
        if show == True:
            plt.show()

        # ---------------------------
        # Write picked points to file
        # ---------------------------
        if len(self.px) > 0:
            with open(output,'wb') as csvfile:
                outfile = csv.writer(csvfile, delimiter=',')
                outfile.writerow(self.px)
                outfile.writerow(self.py)

        return self.fig, self.ax, stream_fft, dimension

    def removeDelay(self, stream):
        '''
        Remove time delay in header.  Time delay may be due to Polytec decoder or GCLAD air-gap.
        '''
        if stream[0].stats.time_delay > 0:
            for trace in stream:
                trim_i = trace.stats.time_delay*1e-6/trace.stats.delta
                trace.data = trace.data[trim_i:]
        else:
            print 'Header time delay value invalid. '
        return stream

    def fkfilter(self, stream, spread=3, dimension='x',colormap='seismic',output='points.csv',show=True):
        '''
        Creates frequency-wavenumber filter of stream data produced by PLACE. 
        Parameters:
             stream : ObsPy stream created with custom header information defined by Scan.py.  Required header information: delta, npts, position.
             spread : size of smoothing average kernel used to smooth the filter edges.
             colormap : colormap to use to display filtered data in the time domain.
             output : filename that selected velocity points are saved to.
        First, FK spectrum is displayed.
             Use left-click to select points.  Each point defines a line with the origin with a slope corresponding to an apparent velocity.  This velocity is displayed when point is chosen.
             To remove a point, right click.  Exactly two points must be selected to define the filter.  
             Press 'enter' when satisfied with the location of the velocity points.  
        Next, the FK filter will be displayed. 
        When FK filter figure is closed, the filtered data is displayed.
        The filtered data figure and axes (to manipulate figure), filtered stream, and the filter (H) is returned.
        '''

        # -------------
        # Plot F-K data
        # -------------
        self.fig, self.ax, stream_fft, dimension = self.fk(stream,dimension=dimension)
        nx = len(stream)
        nt = stream[0].stats.npts
        nk = 2*nx
        nf = 2*nt
        if dimension == 'x':
            dx = stream[1].stats.x_position-stream[0].stats.x_position
        elif dimension == 'theta':
            dx = stream[1].stats.theta_position-stream[0].stats.theta_position
        dt = stream[0].stats.delta

        if (self.py[0]/self.px[0]) > (self.py[1]/self.px[1]):
            vmin = (self.py[1]/self.px[1])*1e3
            vmax = (self.py[0]/self.px[0])*1e3
        else: 
            vmin = (self.py[0]/self.px[0])*1e3
            vmax = (self.py[1]/self.px[1])*1e3

        # create fk filter
        H = np.zeros((nk,nf))
        f = np.arange(-nf/2,nf/2,1)/(nf*dt)
        k = np.arange(-nk/2,nk/2,1)/(nk*dx)

        for i in range(0,nk):
            for j in range(0,nf):
                if vmin < 0 and vmax > 0: # velocity range crosses k = 0
                    if k[i] == 0:
                        H[i,j] = 0
                    else:
                        velocity = (f[j]/k[i])*1e-3
                        if velocity <= vmin and velocity <= 0:
                            H[i,j] = 0
                        elif velocity >= vmax and velocity >= 0:
                            H[i,j] = 0
                        else:
                            H[i,j] = 1
                else:
                    if k[i] == 0:
                        H[i,j] = 1
                    else:
                        velocity = (f[j]/k[i])*1e-3
                        if velocity >= vmin and velocity <= vmax:
                            H[i,j] = 0
                        else:
                            H[i,j] = 1

        t = 1 - np.abs(np.linspace(-1, 1, spread))
        kernel = t.reshape(spread, 1) * t.reshape(1, spread)
        H = scipy.signal.convolve2d(H, kernel, mode='same') #smooth edges of filter

        # show filter
        plt.imshow(np.rot90(H,1),extent=[-1/(2*dx),1/(2*dx),-1e-6/(2*dt),1e-6/(2*dt)],aspect='auto',cmap = 'gray')
        plt.ylim((0,1e-6/(2*dt)))
        plt.ylabel('Frequency (MHz)')
        plt.xlabel('Spatial Frequency (1/mm)')
        plt.show()

        # apply filter to data
        F = stream_fft*H

        # convert back to time domain
        F_ifft = fft.ifft2(fft.fftshift(F))
        filtered_data = F_ifft[0:nx,0:nt]
        filtered_data = np.flipud(np.rot90(filtered_data.real,1))

        filtered_stream = Stream()
        for i in range(0,nx):
            trace = Trace(data=filtered_data[:,i],header=stream[i].stats)
            filtered_stream.append(trace)

        # show filtered data
        self.fig, self.ax, cbar = self.contour(filtered_stream,dimension=dimension,colormap=colormap, show=show)

        return self.fig, self.ax, filtered_stream, H

    def picker(self, event):
        ''' 
        Generic picker for selecting points in a figure.  
        Left-click selects point and plots to figure. 
        Right-click removes point and deletes from figure.
        Returns: 
            px : array of final selected x values.
            py : array of final selected y values.
        '''

        tb = plt.get_current_fig_manager().toolbar 
        if tb.mode == '' and event.inaxes: # checks that toolbar is not in use (e.g. zoom)
            if event.button == 1: # left-click selects point
                self.px.append(event.xdata)
                self.py.append(event.ydata)
                self.ax.plot(event.xdata,event.ydata,'x',color='red')
                plt.draw() 
            elif event.button == 3: # right-click removes point
                del self.px[len(self.px)-1]
                del self.py[len(self.py)-1]
                del self.ax.lines[len(self.ax.lines)-1] 
                plt.draw()
        return self.px, self.py

    def pickV(self, event):
        ''' 
        Picker for selecting points in fk plots.  When selected, the cooresponding velocity is displayed.  
        Left-click selects point and plots to figure. 
        Right-click removes point and deletes from figure.
        Returns: 
            px : array of final selected x values (frequency).
            py : array of final selected y values (wavenumber).
        Returns arrays px and py, the (final) selected x and y values, respectively.
        '''

        tb = plt.get_current_fig_manager().toolbar 
        annotations = []
        if tb.mode == '' and event.inaxes: # checks that toolbar is not in use (e.g. zoom)
            if event.button == 1: # left-click selects point and displays velocity
                self.px.append(event.xdata)
                self.py.append(event.ydata)
                self.ax.plot(event.xdata,event.ydata,'x',color='red')
                v = event.ydata/event.xdata*1e3
                annotations.append(self.ax.annotate(str(int(v))+'m/s',xy=(event.xdata,event.ydata),xytext=None))
                plt.draw() 
            elif event.button == 3: # right-click removes point and displays velocity
                del self.px[len(self.px)-1]
                del self.py[len(self.py)-1]
                del self.ax.lines[len(self.ax.lines)-1] 
                del self.ax.texts[len(annotations)-1]
                plt.draw()
        return self.px, self.py

    def submitV(self,event): # 'enter' submits velocity points
        '''  
        Closes fk spectrum plot for fkfilter() and prints chosen velocity values when 'enter' is pressed on the keyboard and exactly 2 points are selected.
        '''

        if event.key == 'enter':
            if len(self.px) > 2:
                print 'Too many points chosen, choose only two velocity values'
            elif len(self.px) < 2:
                print 'Choose more points! At least two velocity points are needed'
            else:
                for i in range(len(self.px)):
                    print 'velocity ' + str(i+1) + ': ' + str(int((self.py[i]/self.px[i])*1e3)) + ' m/s'
                    plt.close()
