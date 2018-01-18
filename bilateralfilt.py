import numpy as np
import cv2, time

def bilatfilt(I,w=5,sd=1,sr=1):
	dim = I.shape
	Iout= np.zeros(dim)
	#If the window is 5X5 then w = 5	
	wlim = (w-1)/2
	x,y = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
	#Geometric closeness
	c = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*(np.float64(sd)**2)))
	#Photometric Similarity
	Ipad = np.pad(I,(wlim,),'edge')
	for r in xrange(wlim,dim[0]+wlim):
		for c in xrange(wlim,dim[1]+wlim):
			Ix = Ipad[r-wlim:r+wlim+1,c-wlim:c+wlim+1]
			s = np.exp(-np.square(Ix-Ipad[r,c])/(2*(np.float64(sr)**2)))
			k = np.multiply(c,s)
			Iout[r-wlim,c-wlim] = np.sum(np.multiply(k,Ix))/np.sum(k)
	return np.uint8(Iout)

img = cv2.imread('test.jpg')
gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('original img',gimg)
stime = time.time()
fimg = bilatfilt(gimg,w,sd,sr)
print 'Time taken :: '+str(time.time()-stime)+' seconds...'
cv2.imshow('filtered image',fimg)
cv2.waitKey(0)
