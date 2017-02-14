import argparse
import numpy
import pnumpy


parser = argparse.ArgumentParser(description='Compare the effect of different stencils')
parser.add_argument('-n', dest='n', type=int, default=16, 
	                help='Number of cells along each direction')
parser.add_argument('-niter', dest='niter', type=int, default=4, 
	                help='Number of Jacobi iterations')
args = parser.parse_args()

n = args.n
niter = args.niter

def jacobi(inputField, niter=1):
	f = inputField.copy()
	for i in range(niter):
		output = numpy.zeros(f.shape, f.dtype)
		output[0:-1,:] += 0.25 * f[1:,:]
		output[1:,:] += 0.25 * f[0:-1,:]
		output[:, 0:-1] += 0.25 * f[:, 1:]
		output[:, 1:] += 0.25 * f[:, 0:-1]
		# copy
		f[...] = output
	return output

def jacobi2(inputField):
	dc = pnumpy.CubeDecomp(1, inputField.shape)
	op = pnumpy.StencilOperator(dc, periodic=(False, False))
	op.addStencilBranch((+1, 0), 1/5.)
	op.addStencilBranch((-1, 0), 1/5.)
	op.addStencilBranch((0, 1), 1/5.)
	op.addStencilBranch((0, -1), 1/5.)
	op.addStencilBranch((+1,+1), 1/20.)
	op.addStencilBranch((+1,-1), 1/20.)
	op.addStencilBranch((-1,+1), 1/20.)
	op.addStencilBranch((-1,-1), 1/20.)
	return op.apply(inputField)

def trevor(inputField, stencilLen=3):
	nelem = (2 * stencilLen + 1)**2
	fact = 1./nelem
	# self/diagonal
	output = fact * inputField
	for i in range(stencilLen):
		output[0:-stencilLen, :] += fact * inputField[stencilLen:, :]
		output[stencilLen:, :] += fact * inputField[0:-stencilLen, :]
		output[:, 0:-stencilLen] += fact * inputField[:, stencilLen:]
		output[:, stencilLen:] += fact * inputField[:, 0:-stencilLen]
	return output

inputField = numpy.zeros((n, n), numpy.float64)
inputField[n//2-1:n//2+1, n//2-1:n//2+1] = 1.0


from matplotlib import pyplot

"""
tre = trevor(inputField)
f1, a1 = pyplot.subplots()
p1 = pyplot.pcolor(tre) #, vmin=0, vmax=1)
f1.colorbar(p1)
pyplot.title("Trevor's method")
"""

jac = inputField.copy()
for i in range(niter):
	f2, a2 = pyplot.subplots()
	p2 = pyplot.pcolor(jac, cmap='YlOrBr', vmin=0, vmax=1)
	pyplot.title("Jacobi's method")
	pyplot.colorbar(p2)
	jac = jacobi2(jac)


pyplot.show()
