### Requests/issues/notes
 - 2020.09.18 DT label CP triangles, baselines
 - debug/verbose exercising pre nb2 use
 - oifs that run thru amical
 - preferred flags for FringeFitter & fit_fringes
 - remove stale branches
 - make develop branch into main branch in anand0xff/ImPlaneIA 

 scripts/driver_abdor_mir.py:  FringeFitter(... debug=True, ...) fails with   

input: 3D cube  
nrm.core.fit_fringes_single_integration: utils.find_centroid() -> nrm.psf_offset  
Traceback (most recent call last):  
  File "driver_abdor_mir.py", line 162, in <module>  
    verbose=False)
  File "driver_abdor_mir.py", line 107, in main
    verbose=verbose)  
  File "driver_abdor_mir.py", line 83, in analyze_data  
    ff.fit_fringes(fitsimdir+fitsfn)  
  File "/Users/anand/gitsrc/ImPlaneIA/nrm_analysis/nrm_core.py", line 163, in fit_fringes  
    "id":jj},threads)  
  File "/Users/anand/gitsrc/ImPlaneIA/nrm_analysis/nrm_core.py", line 282, in fit_fringes_parallel  
    fit_fringes_single_integration({"object":self, "slc":slc})  
  File "/Users/anand/gitsrc/ImPlaneIA/nrm_analysis/nrm_core.py", line 360, in fit_fringes_single_integration  
    refft = mft.matrix_dft(self.refpsf, 256, 512)  
AttributeError: 'FringeFitter' object has no attribute 'refpsf'  

but works with default (no debug specified)