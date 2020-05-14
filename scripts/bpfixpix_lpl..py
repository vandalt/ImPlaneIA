def bfixpix(data, badmask, n=4, retdat=False):
    """
    Replace pixels flagged as nonzero in a bad-pixel mask with the
    average of their nearest four good neighboring pixels.

    http://www.lpl.arizona.edu/~ianc/python/_modules/nsdata.html#bfixpix

    :INPUTS:
      data : numpy array (two-dimensional)

      badmask : numpy array (same shape as data)

    :OPTIONAL_INPUTS:
      n : int
        number of nearby, good pixels to average over

      retdat : bool
        If True, return an array instead of replacing-in-place and do
        _not_ modify input array `data`.  This is always True if a 1D
        array is input!

    :RETURNS: 
      another numpy array (if retdat is True)

    :TO_DO:
      Implement new approach of Popowicz+2013 (http://arxiv.org/abs/1309.4224)
    """
    # 2010-09-02 11:40 IJC: Created
    #2012-04-05 14:12 IJMC: Added retdat option
    # 2012-04-06 18:51 IJMC: Added a kludgey way to work for 1D inputs
    # 2012-08-09 11:39 IJMC: Now the 'n' option actually works.
    

    if data.ndim==1:
        data = np.tile(data, (3,1))
        badmask = np.tile(badmask, (3,1))
        ret = bfixpix(data, badmask, n=2, retdat=True)
        return ret[1]


    nx, ny = data.shape

    badx, bady = nonzero(badmask)
    nbad = len(badx)

    if retdat:
        data = array(data, copy=True)
    
    for ii in range(nbad):
        thisloc = badx[ii], bady[ii]
        rad = 0
        numNearbyGoodPixels = 0

        while numNearbyGoodPixels<n:
            rad += 1
            xmin = max(0, badx[ii]-rad)
            xmax = min(nx, badx[ii]+rad)
            ymin = max(0, bady[ii]-rad)
            ymax = min(ny, bady[ii]+rad)
            x = arange(nx)[xmin:xmax+1]
            y = arange(ny)[ymin:ymax+1]
            yy,xx = meshgrid(y,x)
            #print ii, rad, xmin, xmax, ymin, ymax, badmask.shape
            
            rr = abs(xx + 1j*yy) * (1. - badmask[xmin:xmax+1,ymin:ymax+1])
            numNearbyGoodPixels = (rr>0).sum()
        
        closestDistances = unique(sort(rr[rr>0])[0:n])
        numDistances = len(closestDistances)
        localSum = 0.
        localDenominator = 0.
        for jj in range(numDistances):
            localSum += data[xmin:xmax+1,ymin:ymax+1][rr==closestDistances[jj]].sum()
            localDenominator += (rr==closestDistances[jj]).sum()

        #print badx[ii], bady[ii], 1.0 * localSum / localDenominator, data[xmin:xmax+1,ymin:ymax+1]
        data[badx[ii], bady[ii]] = 1.0 * localSum / localDenominator

    if retdat:
        ret = data
    else:
        ret = None

    return ret
