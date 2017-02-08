import pnumpy
import numpy
from mpi4py import MPI
import argparse
import sys
import netCDF4
from matplotlib import pylab
import time

parser = argparse.ArgumentParser(description='Compute the probability of precipitation')
parser.add_argument('--file', type=str, dest='file', 
                    default='total_precip.nc',
                    help='File containing total_precip data')
parser.add_argument('--threshold', type=float, dest='threshold', default=8.33333333333333e-07,
                    help='Min amount of rain',)
parser.add_argument('--niter', type=int, dest='niter', default=1, 
                    help='Number of iterations')
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot')

args = parser.parse_args()

def plotProb(lats, lons, data, title):
    from mpl_toolkits.basemap import Basemap

    # plot the data
    f = pylab.figure()
    p = pylab.pcolor(lons, lats, data, vmin=0, vmax=1)
    pylab.colorbar(p)
    # add the basemap
    mp = Basemap(llcrnrlon=160, urcrnrlon=185, llcrnrlat=-49, urcrnrlat=-30, resolution='h')
    mp.drawcoastlines(color='w')
    f.savefig('prob_precip_pe{}.png'.format(pe))

nprocs = MPI.COMM_WORLD.Get_size()
pe = MPI.COMM_WORLD.Get_rank()

f = netCDF4.Dataset(args.file)
size = f.variables['total_precip'].shape[1:]


# domain decomp
dc = pnumpy.CubeDecomp(nprocs, size)
if not dc.getDecomp():
    print('ERROR: cannot find a valid decomposition for {} procs and size {}'.format(nprocs, size))
    sys.exit(1)

# local start/end grid indices
slab = dc.getSlab(pe)
ibeg, iend = slab[0].start, slab[0].stop
jbeg, jend = slab[1].start, slab[1].stop

# read the local data
total_precip = f.variables['total_precip'][0, ibeg:iend, jbeg:jend] # first time step
lats = f.variables['latitude'][ibeg:iend, jbeg:jend]
lons = f.variables['longitude'][ibeg:iend, jbeg:jend]

f.close()

# create averager
op = pnumpy.StencilOperator(dc, periodic=(False, False))
op.addStencilBranch((+1,  0), 1.0/4.0)
op.addStencilBranch(( 0, +1), 1.0/4.0)
op.addStencilBranch((-1,  0), 1.0/4.0)
op.addStencilBranch(( 0, -1), 1.0/4.0)

"""
if args.plot:
    # plot entire data
    f = pylab.figure()
    p = pylab.pcolor(lons, lats, cube.data)
    pylab.title('precip kg/m^2')
    f.savefig('precip.png')
"""



# apply
totaltime = 0
precip = numpy.array(total_precip > args.threshold, numpy.float64)
for i in range(args.niter):
    t0 = time.time()
    out = op.apply(precip)
    precip[...] = out
    t1 = time.time()
    totaltime += t1 - t0

# compute global checksums and time
totaltime /= float(nprocs) # average
totaltime0 = numpy.sum(MPI.COMM_WORLD.gather(totaltime, 0))
checksum = out.sum()
checksum0 = numpy.sum(MPI.COMM_WORLD.gather(checksum, 0))
if pe == 0: 
    print('i = {} checksum: {} time: {}'.format(i, checksum0, totaltime0))

# plot data
if args.plot: 
    plotProb(out, 'prob precip')

