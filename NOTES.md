# nrm_analysis 

#### Delivery Notes 2020/05

Reduces aperture masking images to fringe observables, calibrates, does basic model fitting. Package development led by Alexandra Greenbaum following legacy code by Greenbaum, Anand Sivaramakrishnan, and Laurent Pueyo. Contributions from Sivaramakrishnan, Deepashri Thatte, Johannes Sahlmann, Anthony Soulain.

## ImPlaneIA
AMI data is affected by three parameters, in addition to the usual noise sources.  The parameters are pistons (7 optical path delays between mask holes, usually with zero mean), psf offsets, and pupil rotation (which is reflected in rotation of the image).  ImPlaneIA finds these values and creates an analytical model using the psf offsets and rotation.  Pistons are not yet used when creating a model, though the parameter is passed to the routine. ImPlaneIA can also accept user supplied values of psf offsets and rotation or it utilizes a combination of user supplied values and parameters it measures itself. 

The coefficients obtained from the least-squares fit (between the model and data) are used to calculate fringe phases, fringe amplitudes, closure phases and closure amplitudes 

**scripts/implaneia_driver.py** is used to simulate and analyze data. Calling **simulate_data()** in this driver is optional-simulate new data, or use the file **all\_effects\_data\_mir.fits** in the analysis.

Observables are extracted with a call to **analyze_data()**:

		_aff, _psf_offset_r, _psf_offset_ff, fringepistons = analyze_data(df,             
                      affine2d=None, psf_offset_find_rotation = (0.0,0.0),  
                      psf_offset_ff = None, rotsearch_d=_rotsearch_d, 
                      fringepistons_ff=None)

* **df** data file name
* **affine2d** is user supplied or measured Affine2d object. A user supplied affine2d is created with a rotation value (**rot** in **implaneia_driver.py**).
* **psf\_offset\_find\_rotation** are the psf offsets used to find rotation of data. Finding rotation involves simulating data at rotation values specified by rotsearch_d and then cross-corelating with data to create an affine object that is used for fringefitting. Use an initial guess of (0,0) or a known psf offset for this parameter.
* **psf\_offset\_ff** is the psf offset used in fringefitting to create the model array. ImPlaneIA uses the center of the brightest pixel as the coordinate system to calculate psf offsets for fringefitting. When doing the fringefitting the data is trimmed such that the brightest pixel is at the center of the trimmed image. **psf\_offset\_ff** is the measured or user supplied value from the center of this trimmed image that has odd dimensions.
* **rotsearch\_d** is the set of values used for rotation search. This is used in conjunction with **psf\_offset\_find\_rotation** to create an affine object that is used for fringefitting; they not used when affine2d is user supplied.



**ami_analyze.cfg file/parameters file**

		Oversample = 3
		set_affine_rotation = None (default) or e.g. 2.0
		set_psf_offset_find_rotation = (0.0,0.0)  (default ) or e.g. (0.48,0.0)
		set_rotsearch_d = np.arange(-3, 3.1, 1)  (default)  
							  or e.g. (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
							  (Minimum of three values: finds quadratic maximum)
		set_psf_offset_ff = None (default) or e.g. (-0.52, 0.0)
		set_fringepistons_ff = None (default) or an array of seven numbers 
		
**Documentation for the parameter file** (draft):

If set\_affine\_rotation is a number, it is used for the rotation of analytical model in degrees.

If set\_affine\_rotation is None, the user provides rosearch\_d in degrees and set\_psf\_offset\_find\_rotation in detector pixel units.  These two values are used to find rotation and use it to create the analytical model. The latter two parameters are not used when 
set\_affine\_rotation is a number.

Implaneia uses the center of the brightest pixel as the coordinate system to calculate psf offsets (set\_psf\_offset\_ff) for fringefitting. **_This coordinate system may or may not match the coordinate system of set\_psf\_offset\_find\_rotation_**.

**analyze\_data** function definition

	def analyze_data(df, affine2d=None, 
				       psf_offset_find_rotation = (0.0,0.0), psf_offset_ff = None, 
				       rotsearch_d=None, set_pistons=None)

Example of values for parameters (these are used in implaneia_driver.py for odd-sized fields of view):

    args_odd_fov = [[None, (0.0,0.0), None, _rotsearch_d ,np.zeros((7,))],
                              [None, (0.0,0.0), (-0.5199,0.0),_rotsearch_d, np.zeros((7,))],
                              [None, (0.48,0.0), None, _rotsearch_d, np.zeros((7,))],
                              [aff,(0.0,0.0),None, None, np.zeros((7,))],
                              [None, (0.48,0.0), (-0.5199,0.0),_rotsearch_d, np.zeros((7,))],
                              [aff, (0.48,0.0), (-0.5199,0.0), _rotsearch_d, np.zeros((7,)),],
                              ]

    
Note: 
1. aff in the above is an Affine2d object instantiated as a pure rotation of **rot**.  
2. analyze\_data() does not use rotsearch\_d and psf\_offset\_find\_rotation when its **affine2d** parameter is assigned an Affine2d instance (instead of None).  If **affine2d=None**, rotsearch\_d and psf\_offset\_find\_rotation are used to find the pupil rotation using the data.  
3. Use of **fringepistons\_ff** is not yet implemented.  When it is, the set of fringe pistons (one for each hole in the mask) can be used to create the **model** array of fringes, total flux, and the DC level (bias) of the image. Currently only **psf\_offset\_ff** and **affine2d** are used to make the model.
