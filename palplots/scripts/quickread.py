#!/usr/bin/env python
'''
Quick-read function for data acquired with PLACE.  

Options:
-n --name=
    Filename.  Include .h5 prefix.
    Example: -n S8-rot
--axis 
    Axis from header to sort and plot by. (x, theta, etc.)
    Example: --axis theta
--plot
    Type of plot to use (wiggle or contour)
    Example: --plot contour

Example usage:
quickread -n S8-rotd-postreaction --axis theta --plot wiggle
'''

from __future__ import print_function
from .. import Plot
from obspy.core import read
import getopt
import sys

def main():
    try: 
        opts, args = getopt.getopt(sys.argv[1:],'h:n:a:p',['help','name=','axis=','plot='])
    except getopt.error as msg:
        print(msg)
        print('for help use --help')
        sys.exit(2)

    for o, a in opts:
        if o in ('-h','--help'):
            print(__doc__)
            sys.exit()
        if o in ('-n','--name'):
            filename = str(a)
        if o in ('-p', '--plot'):
            plot = str(a)
        if o in ('-a', '--axis'):
            dimension = str(a)

    # read data
    stream = read(filename,'H5',apply_calib=True)

    # sort by appropriate dimension
    sorter = dimension+'_position'
    stream.sort(keys=[dimension+'_position'])
    stream.detrend()

    # plot
    if plot == 'contour':
        Plot().contour(stream,dimension=dimension)
    elif plot == 'wiggle':
        Plot().wiggle(stream,dimension=dimension)
    else:
        print('invalid plot choice')

if __name__ == '__main__':
    main()
