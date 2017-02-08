numCores = [1, 2, 4, 8]
times = [65.004, 30.196, 19.587, 17.454]

speedup = [times[0]/t for t in times]

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


from matplotlib import pylab
pylab.plot(numCores, speedup, numCores, speedup, 'ko')
pylab.title('speedup')
pylab.xlabel('number of cores')
pylab.plot(numCores, numCores, 'k--')
pylab.axis([1, 8, 1, 4])
pylab.show()

