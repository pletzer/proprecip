import iris
import pnumpy
import numpy
from mpi4py import MPI
import argparse
import sys
import netCDF4
from matplotlib import pylab
import time

iris.FUTURE.netcdf_promote = True

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

nprocs = MPI.COMM_WORLD.Get_size()
pe = MPI.COMM_WORLD.Get_rank()

# read the data
cube = iris.load(args.file, iris.Constraint(cube_func = lambda c: c.var_name == 'total_precip'))[0]
f = netCDF4.Dataset(args.file)
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
f.close()

# domain decomp
size = cube.data.shape[1:] # first dimension is time
dc = pnumpy.CubeDecomp(nprocs, size)
if not dc.getDecomp():
    print('ERROR: cannot find a valid decomposition for {} procs and size {}'.format(nprocs, size))
    sys.exit(1)

ndims = dc.getNumDims()
# local start/end grid indices
slab = dc.getSlab(pe)
ibeg, iend = slab[0].start, slab[0].stop
jbeg, jend = slab[1].start, slab[1].stop

# average the data 
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

print(lats.min())
print(lats.max())
print(lons.min())
print(lons.max())

lats = lats[ibeg:iend, jbeg:jend]
lons = lons[ibeg:iend, jbeg:jend]

def plotProb(data, title, index):
    from mpl_toolkits.basemap import Basemap

    # plot the data
    f = pylab.figure()
    p = pylab.pcolor(lons, lats, data, vmin=0, vmax=1)
    pylab.colorbar(p)
    # add the basemap
    mp = Basemap(llcrnrlon=160, urcrnrlon=185, llcrnrlat=-49, urcrnrlat=-30, resolution='h')
    mp.drawcoastlines(color='w')
    f.savefig('prob_precip_pe{}_index{}.png'.format(pe, index))

# apply 
if pe == 0: print('applying stencil..')
precip = numpy.array(cube.data[0, ibeg:iend, jbeg:jend] > args.threshold, numpy.float64)
for i in range(args.niter):
    t0 = time.time()
    out = op.apply(precip)
    precip[...] = out
    # compute global checksums
    checksum = out.sum()
    checksum0 = numpy.sum(MPI.COMM_WORLD.gather(checksum, 0))
    t1 = time.time()
    if pe == 0: 
        print('i = {} checksum: {} time: {}'.format(i, checksum0, t1 - t0))
    # plot data
    if args.plot: 
        plotProb(out, 'prob precip i = {}'.format(i), i)
if pe == 0: print('done.')

