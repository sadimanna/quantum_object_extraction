import numpy as np
import scipy.sparse as sp
import time, datetime

dimr = (320,480)
ld = dimr[0]*dimr[1]
stime = time.time()
now = datetime.datetime.now()

ham_kin = sp.coo_matrix(([],([],[])),shape=(0,ld))
batchsize = 64
nbatch = ld/batchsize
for nb in xrange(nbatch):
	print 'Batch...'+str(nb+1)

	row4 = np.array([])
	col4 = np.array([])
	row1 = np.array([])
	col1 = np.array([])

	for i in xrange(batchsize*nb,batchsize*(nb+1)):
		for j in xrange(ld):
			if i==j:
				row4 = np.append(row4,i)
				col4 = np.append(col4,j)
				if j != 0 and j%dimr[1]!=0:
					row1 = np.append(row1,i)
					col1 = np.append(col1,j-1)
				if j != ld-1 and (j+1)%(dimr[1])!=0:
					row1 = np.append(row1,i)
					col1 = np.append(col1,j+1)
				if j-dimr[1]>=0:
					row1 = np.append(row1,i)
					col1 = np.append(col1,j-dimr[1])
				if j+dimr[1]<=ld-1:
					row1 = np.append(row1,i)
					col1 = np.append(col1,j+dimr[1])
	if nb>=1:
		row1 = row1 - batchsize*nb
		row4 = row4 - batchsize*nb

	coomat1 = sp.coo_matrix((np.array(len(row1)*[-1]),(row1,col1)),shape=(batchsize,ld))
	coomat4 = sp.coo_matrix((np.array(len(row4)*[ 4]),(row4,col4)),shape=(batchsize,ld))

	if ham_kin.shape[0] == 0:
		ham_kin = coomat1 + coomat4

	else:
		ham_kin = sp.vstack([ham_kin,coomat1 + coomat4])

print ham_kin.shape
#print ham_kin.toarray()

sp.save_npz('ham_kin.npz',ham_kin)

print str(time.time()-stime)+'seconds...'
print 'Starting Time:: ',now
print 'Ending Time:: ',datetime.datetime.now()
