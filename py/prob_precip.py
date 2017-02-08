import iris
import pnumpy
import numpy
from mpi4py import MPI
import argparse
import sys
import netCDF4

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

# average the data 
op = pnumpy.StencilOperator(dc, periodic=(False, False))
op.addStencilBranch((+1,  0), 1.0/9.0)
op.addStencilBranch((+1, +1), 1.0/9.0)
op.addStencilBranch(( 0, +1), 1.0/9.0)
op.addStencilBranch((-1, +1), 1.0/9.0)
op.addStencilBranch((-1,  0), 1.0/9.0)
op.addStencilBranch((-1, -1), 1.0/9.0)
op.addStencilBranch(( 0, -1), 1.0/9.0)
op.addStencilBranch((+1, -1), 1.0/9.0)
op.addStencilBranch(( 0,  0), 1.0/9.0)

# apply 
print('applying stencil..')
precip = numpy.array(cube.data[0,...] > args.threshold, numpy.float64)
for i in range(10):
    out = op.apply(precip)
    precip = out
print('done.')

# plot
if args.plot:
    from matplotlib import pylab

    coords = cube.coords()

    print(lats.shape)
    print(lons.shape)
    print(out.shape)

    pylab.figure(1)
    data = cube.data[0,...]
    print(data.shape)   
    p1 = pylab.pcolor(lons, lats, data)
    pylab.title('total_precip kg/m^2')
    pylab.colorbar(p1)

    pylab.figure(2)
    p2 = pylab.pcolor(lons, lats, out, vmin=0, vmax=1)
    pylab.colorbar(p2)
    pylab.title('probability')
    pylab.show()

