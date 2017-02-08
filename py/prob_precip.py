import iris
import pnumpy
from mpi4py import MPI
import argparse
import sys

iris.FUTURE.netcdf_promote = True

parser = argparse.ArgumentParser(description='Compute the probability of precipitation')
parser.add_argument('--file', type=str, dest='file', 
                    default='total_precip.nc',
                    help='File containing total_precip data')
parser.add_argument('--threshold', type=float, dest='threshold', default=8.33333333333333e-07,
                    help='Min amount of rain',)
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot')

args = parser.parse_args()

nprocs = MPI.COMM_WORLD.Get_size()
pe = MPI.COMM_WORLD.Get_rank()

# read the data
cube = iris.load(args.file, iris.Constraint(cube_func = lambda c: c.var_name == 'total_precip'))[0]

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
out = op.apply(cube.data[0,...])
print('done.')

# plot
if args.plot:
    from matplotlib import pylab
    coords = cube.coords()
    #print(coords)
    #yy = coords[0].points
    #xx = coords[1].points
    #print(cube.data.shape)
    #print(yy.shape)
    pylab.figure(1)
    data = cube.data[0,...]
    print(data.shape)
    pylab.pcolor(data)
    pylab.figure(2)
    pylab.pcolor(out)
    pylab.show()

