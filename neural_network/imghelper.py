import numpy as np

def make_grid(nrows,ncols,rpx,cpx,arr):
    """
    Takes a array[:,rpx*cpx] of image pixels and returns back an image grid.
    nrows - number of rows of images
    ncols - number of columns of images
    rpx - number of horizontal pixels in each image
    cpx - number of vertical pixels in each image
    arr - two dimensional array with n rows of (rpx*cpx) numbers each
    """
    arr_images = arr.reshape(nrows,ncols,rpx,cpx)
    for r in range(nrows):
      for c in range(ncols):
          arr_images[r,c] = arr_images[r,c].T
    image = np.empty((nrows*rpx,ncols*cpx))
    for r in np.arange(nrows):
      for ir in np.arange(rpx):
          pline = np.array([])
          for c in np.arange(ncols):
              pline = np.append(pline,arr_images[r,c,ir])
          image[r*rpx+ir] = pline
    return image

