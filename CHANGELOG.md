# Changelog
All notable changes to this project will be documented in this file.

The format is based on: 
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to: 
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

***
## [1.7.5] - 2019-05-14 - RIHY

### Fixed

* Commented-out sections of Train Analysis example script resurrected. This 
  involved updating the `collate_stats()` method of the `Multiple()` class, 
  to make use of updated internal methods for handling results from 
  time-stepping analyses. Needs some further work to handle the case of systems 
  comprising multiple sub-systems however, as indicated by comments in the code.


## [1.7.4] - 2019-05-02 - RIHY

### Fixed

* Bug when running analysis for SDOF modal systems fixed

* Issue with automatic forcing of ylim in `freq_response_results.plot()` fixed 
  (this had been causing plots to go off-scale previously)

## [1.7.3] - 2019-04-05 - RIHY

### Fixed

* Error with `EigResults()` plot method #4; damping ratio now correctly 
  expressed as a percentage

### Added

* Absolute displacement output added by default for TMDs

### Changed

* Improvements to plotting in `freq_response_rslts.py` and `eig_results.py`

## [1.7.2] - 2019-03-21 - RIHY

### Fixed

* Fixed bugs in `modalsys.py` for case of system with only 1 mode

* Improved plotting in `freq_response_rslts.py`

### Added

* TMD example script added


## [1.7.1] - 2019-03-21 - RIHY

### Fixed

* Hotfix to `check_class()` method in `common.py`


## [1.7.0] - 2019-03-21 - RIHY

### Changed

* Previous module `wind.py` split across three new modules:

    * `wind_response.py` houses functions/classes concerned with calculation 
      of response to wind loading
      
    * `wind_env.py` houses functions/classes concerned with defining wind 
      environments
      
    * `wind_misc.py` houses all other functions not allocated to the above. 
      The intention is that these will be sorted-through and relocated / 
      deleted eventually
      
### Added

* New module `hysteresis.py`, together with related example, to provide 
  classes to model hysteretic behaviour

* New module `eig_results.py`, with class to act as container for 
  eigenvalue results, to replace previous dict usage. Coding is intended to be  
  backwards-compatible, but allows eigenproperties plot methods previously 
  housed as `DynSys()` class methods to more logically be implemented as 
  methods associated with the new `Eig_Results()` class.
  
* Note with this change, the `Eig_Results()` class only holds unique data, i.e. 
  eigenvalues and left- and right-eigenvectors. All other eigenproperties are 
  calculated upon request. This is intended to be more robust and will ensure 
  consistent properties are always obtained.
  
* Lots of alternative getter methods defined for `Eig_Results()` to be more 
  flexible with spellings / names when requesting certain attributes
  
* New module `freq_response_results.py`, with class to act as container 
  for frequency response results, to replace previous dict usage. Coding is 
  backwards-compatible, except `PlotFrequencyResponse()` method previously 
  implemented within `dynsys.py` now used as `FreqResponse_Results.plot()` 
  method
  
* New module `mesh.py`, to faciliate definition or and manipulation of meshes
  (i.e. interconnected nodes and elements)
  
* New module `nodle.py`, to provide methods for importing data from NODLE 
  Excel input files (COO and MEM data) and .res files (DIS data)

***

## [1.6.2] - 2018-10-19 - RIHY

### Added

* Option added to `LatSync_McRobie.__init__()` method, to allow system matrices 
  determined at each Np value considered to be stored for subsequent usage

* Verbose output control added to `LatSync_McRobie()` class methods

* `LatSync_McRobie().calc_Np_crit()`: parameters / key results at 
  Np=Np_crit instability point extracted and held as attributes
  
* New module `wind_section.py` added to define wind sections / resistances


## [1.6.1] - 2018-10-18 - RIHY

### Fixed

* Default mp(f) curve corrected; had previously mis-interpeted Fig 9(b) of 
  MacDonald's paper


## [1.6.0] - 2018-10-18 - RIHY

### Changed

* 'LatSync_McRobie' analysis extended to allow for frequency-dependent 
  damping and added mass effect of pedestrians, with default curves defined 
  based largely on John MacDonalds _'Lateral excitation of bridges by balancing 
  pedestrians'_ paper, which has been included in the 'references' section.

### Added

* `AnimateResults()` method updated to plot loading as well as deformed system

* Lateral loading model in the spirit of UK NA to BS EN 1991-2 added

### Fixed

* Minor bug fixed, relating to selection of correct S_eff for mode

***

## [1.5.1] - 2018-09-20 - RIHY

### Changed

* `AnimateResults()` API updated to allow optional keywords to be passed down to
  methods called within, e.g. to customise plots/animations produced

* Train analysis example updated to illustrate usage of `AnimateResults()` with 
  animation customisation


## [1.5.0] - 2018-09-19 - RIHY

### Added

* `AnimateResults()` method implemented properly (in a generalised fashion) for 
  the first time. System-specific plot methods defined for common system types

* `PlotEnergyResults()` and related calculation methods added to `TStep_results`
  class

***

## [1.4.2] - 2018-09-17 - RIHY

### Changed

* Improvements to how `event_funcs` and `post_event_funcs` are implemented 
  within `TStep()` class of `tstep.py`

* Bounce sim example implemented (in conjunction with the above code 
  updates)

* `CalcFreqResponse()` method and function interface revised to give code 
  that is better structured and more generally applicable.

* Minor improvements to lat sync analysis:
    * Analysis is extended in case of not detecting net zero damping within range 
      of pedestrian numbers originally specified
    * User control over verbose output added

### Added

* `UKNA_BSEN1991_2_Figure_NA_11()` function added to define Figure NA.11 in 
  UK NA to BS EN 1991-2:2003.

### Fixed

* Correction to Figure NA.9 in UK NA to BS EN 1991-2, as defined in PD6688-2
  has been implemented. Option added to select values according to each 
  standard, but default behaviour is to use the PD6688-2 correction.


## [1.4.1] - 2018-07-23 - RIHY

### Added

* `calc_Np_crit()` method added to `LatSync_McRobie()` class, to allow 
  critical number of pedestrians to cause lateral instability to be
  considered. This method is also now called by LatSync_McRobie().run().

* Improvements to related plotting routines; Np_crit overlaid onto plots


## [1.4.0] - 2018-07-23 - RIHY

### Added

* Check added as to Scipy version when importing `tstep.py` module; v1.0 or 
  greater required (an exception will be raised otherwise, as `solve_ivp()` 
  method unavaliable in earlier versions.

* `scipy.linalg.null_space()` method now used when possible (i.e. when Scipy 
  v1.1 or greater is used), to save duplication of code in `null_space()` 
  method with `dynsys.py` module. In the future the plan will be to use the 
  Scipy method alone, to allow reduction in code.

* `LatSync_McRobie()` class defined within `ped_dyn.py` module. Implements 
  lateral synchorous vibration calculations per McRobie's Stockton on Tees 
  footbridge paper (refer `/references`). See also validation script, which 
  obtains excellent agreement with figures presented in McRobie's paper.

***

## [1.3.0] - 2018-07-20 - RIHY

### Added

* New method `transform_to_unconstrained()` defined in `dynsys.py` module, to 
  implement method for transforming a constrained system (i.e. a system with 
  constraint equations) into an equivilent unconstrained one. [This is a key 
  step in calculations such as complex eigensolution and evaluation of 
  frequency transfer matrix.]. This method can be generally used - but the 
  main motivation behind its implementation is to allow code reuse in multiple 
  methods of the core `DynSys` class, e.g. `DynSys.CalcEigenproperties()`.
  
* `DynSys.CalcEigenproperties()` method will now work for systems with 
  constraints. Accuracy of the implementation has been validated by extension 
  of Warburton TMD validation script.

***

## [1.2.2] - 2018-07-19 - RIHY

### Fixed

* Further fix associated with lambda fix at v1.2.1: when multiple modes 
  included array of maximum modeshape ordinate now calculated and appropriate 
  value used in calculations targeting specific mode. Previous behaviour was to 
  obtain maximum ordinate across all modes, which is inaccurate when mode being 
  considered does not have this ordinate.
  
* Minor fixes to iron-out runtime bugs in routines associated with steady-state 
  crowd loading analysis. Bugs relates to plotting and results reporting, i.e. 
  no impact on the accuracy of calculations made using previous versions.


## [1.2.1] - 2018-07-17 - RIHY

### Added

* `damper.py` module added; `TMD()` class defined within this module, to allow 
  tuned mass damper systems to be conveniently defined

### Fixed

* Bug associated with calculation of lambda factor for UK NA steady-state crowd 
  loading analysis fixed.


## [1.2.0] - 2018-06-12 - RIHY

### Changed

* Pedestrian dynamics classes and functions moved to new module `ped_dyn.py`
  (previously these were within the `dyn_analysis.py` module)

* Many method previously implemented as class methods of `DynSys` class now 
  re-implemented as functions within the `dynsys.py` module. Class methods 
  still exist as per previous versions, but these use the public functions to 
  do the work
  
* Various misc updates to plotting methods within `tstep_results.py` module

* The majority of docstrings have been reviewed/improved, to ensure they 
  reflect the source code and provide more references to theory, codes etc.

### Added

* `tutorial.ipynb`, a IPython (Jupyter) notebook which provides an introduction 
  on how to go about using the classes/functions within the package

* Crowd loading to UK NA to BS EN 1991-2 now implemented

* `CalcEigenproperties()` method of `DynSys` class reimplemented to handle 
  systems with constraints. Approach for unconstrained systems similar to 
  previous versions except `scipy.linalg.eig()` method now used.
  
* `/tests` updated: new test of `CalcEigenvalues()` method devised

### Fixed

* Bug as noted by BNCY in `dyn_analysis.py` now fixed. A check of the class 
  name of `loadtrain_obj` was being made, but as a result of recent changes 
  this was checking the derived class name.

* `trainAnalysis.py` updated with minor change to reflect new `stats_dict` 
  nested dict structure
  
* Bug associated with output matrices fixed; was only an issue when re-defining 
  systems

***

## [1.1.1] - 2018-05-22 - RIHY

_Hotfix responding to BNCY's 21/05/2018 email_

### Fixed

* Tempororary fix implemented to address bug, by commenting-out rows.
  To be addressed properly in future releases.
  

## [1.1.0] - 2018-04-23 - RIHY

### Changed
* Version number now included in html docs

### Added
* New functionality to implement UK NA to BS EN 1992-1 footbridge dynamics:
    * New class in `loading.py`, `UKNA_BSEN1991_2_walkers_joggers_loading`.
      Defines moving load group for walkers/joggers. Inherits most of the 
      functionality from existing `LoadTrain` class.
    * Figure functions, to return parameters given by code figures:
        * `UKNA_BSEN1991_2_Figure_NA_8` in `loading.py`
        * `UKNA_BSEN1991_2_Figure_NA_9` in `dyn_analysis.py`
    * New class in `dyn_analysis.py`, `UKNA_BSEN1991_2_walkers_joggers` (this 
      inherits functionality from existing `MovingLoadAnalysis` class)
    * New class in `dyn_analysis.py`, `PedestrianDynamics_transientAnalyses`.
      Inherits functionality from existing `Multiple` class. This can be used 
      to carry out all the necessary walking/jogging analyses for all modes.
    * Example added to `examples\Pedestrian dynamics\UK NA walkers joggers` to 
      test and illustrate the use of these number classes and functions   

***

## [1.0.0] - 2018-04-17 - RIHY

### Changed

* `warburton_TMD.py` validation script moved to subfolder. In future all 
  validation scripts should be kept in subfolders to avoid confusion / 
  cross-over of inputs
  
* System matrices (e.g. M_mtrx, K_mtrx etc) now intended to be held as private 
  attributes; renames _M_mtrx, _C_mtrx to denote this change in intent. 
  `GetSystemMatrices()` member function of `DynSys` class re-instated (this 
  was previously marked as deprecated) and overhauled to act as the primary 
  get() function for obtaining system matrices
  
* Significant non backwards-compatible overhaul of how constraints are defined 
  and handled internally within the `DynSys` class. Constraint matrices are 
  now held within a dict and kept local to each dynamic system. This allows for 
  straightforward addition/removal of constraints and improves modularity. 
  
* Improved functionality for systems with sub-systems appended using 
  `AppendSystem()` method
  
* Plotting methods and response / response stats calculation methods revised to 
  be more structured and cater for systems with multiple subsystems
  
* `Graphviz` package used to produce call graphs for train analysis example 
  script. This type of testing was found to be very assistive in diagnosing 
  slow steps in the analysis and allowed overall runtime to be dramatically 
  improved. Tests like this should be carried out periodically in the future. 
  Dedicted subfolder made in /tests folder for this purpose.
   
### Added

* Validation example added to verify accuracy or revised ResponseSpectrum()
  function. Example computes response spectra for the classic El-Centro (1940)
  earthquake and makes comparison against published plots. Excellent agreement 
  found!
  
* New method `PrintResponseStats()` added to `tstep_results` class, to allow 
  response statistics to be printed to text window in a nice manner

### Fixed

* Bug fix for ResponseSpectrum() function in `dyn_analysis`. 
  Equation of motion corrected to be -M.a (minus sign previously omitted)
  Acceleration spectrum now based on _absolute_ acceleration, as is conventional
  (previous versions omitted to add the input ground motion to the SDOF responses 
  calculated)
  
### Removed

* GetRealWorldDOFs() method removed from DynSys class (this had been marked as 
  deprecated since v0.1.0)

***

## [0.5.1] - 2018-03-19 - RIHY

### Changed

* `docs` remade (forgot to do this before releasing v0.5.0)

## [0.5.0] - 2018-03-19 - RIHY

### Added

* `ResponseSpectrum` function added to `dyn_analysis` module, to allow 
  calculation of SDOF response spectrum (per seismic analysis) based on ground 
  motion acceleration time series

### Fixed

* Bugs in plotting routines within `tstep_results` arising from change to 
  response results internal storage structure
* Miscellaneous fixes associated with `None` optional arguments

### Changed

* Statistics calculated using `CalcResponseStats()` within `tstep_results` 
  module are now returned as a dict, rather than list of dicts

### Removed

***

## [0.4.0] - 2018-03-19 - RIHY

### Added

* `tstep_results.WriteResults2File()` method allows time-series results to 
  be written to file. Linking to this function made in `tstep` and 
  `dyn_analysis` class methods. Files exported have QA-relevant headers (date, 
  version etc)
  
* `dyn_analysis.Dyn_Analysis`and `dyn_analysis.Multiple` objects can now be 
  saved and loaded using `dill` module (similar to `pickle`). 
  This allows objects (incl results) to be stored as binary files and 
  reloaded later. This is very useful when carrying out lots of analyses, as 
  it allows all data to be saved and reloaded later e.g. for further 
  manuipluation
  
* Optional argument `fLimit` added to `modalsys.ModalSys()`, to allow maximum 
  modal natural frequency to be specified. Modes with fn > fLimit will not be 
  included in analysis (a warning message is printed to show this)
  
* `DynSys.PlotResponsePSD()` added to implement plotting of PSDs of response 
  time series
* Git version control used now

### Fixed

* Code hide x_axis within `tstep_results.PlotResponseResults()` function 
  removed as obsolete: subplots now use `sharex=True`

***

## [0.3.0] - 2018-03-16 - RIHY

### Added

* In `DynSys.__init__()` option added to represent system matrices as sparse 
  matrices (which they usually are), using `scipy.sparse.csc_matrix` format. 
  Code updates implemented in all modules to handle sparse matrices. However 
  current dense methods are used by default as this is regarded as a work in 
  progress
  
* New module `tstep_analysis` added. This modules contains classes to implement 
  certain forms of time-history analysis
  
* `setup.py` created, per PyPi guidance

* Within `dynsys` and derived classes, output matrix and corresponding names no 
  longer stored as a list of matrices / list of names, as this was proving 
  unnecessarily confusing! Corrections made in all modules (I think...)

### Fixed

* When `modalsys.AppendTMDs` is used, this has the effect of increasing the 
  nDOF of system. Need therefore to adjust structure of `output_mtrx`. This is 
  now done

### Changed

* If `J_mtrx=None` is passed into DynSys() __init__ function then `J_mtrx` is 
  now initialised as an array of shape [0,nDOF]
  
* Significant overhaul of `tstep` and `tstep_results`

* Package version now in `__init__.py` as single point of reference

* Treatment of response / output matrices in `tstep_results`, `tstep` and 
  `dynsys` harmonised
  
* Miscellaneous improvements / options added to results plotting routines

***

## [0.2.0] - 2018-03-02 - RIHY

### Added

* `/tests/tests_basic.py`, to contain test functions

* `/tests/test_docs.py`, to auto-generate documentation concerning tests
* SDOF properties conversion functions added to `dynsys` for common use in 
  other modules
  
* `tstep` added to package, to provide time-stepping analysis functionality. 
  This was previously a seperate package, but given inter-dependecies it makes 
  more sense for this to be part of `dynsys`

### Changed

* Package folder structure extended:

    * `validation` subfolder created, to contain scripts which are more 
    involved than simple `unittest` routines and serve to validate the library 
    by comparing results obtained using `dynsys` routines against published 
    results given in classical dynamics literature
    
    * `docs`, `img`, `ref` subfolders created in `tests`

***

## [0.1.0] - 2018-02-28 - RIHY

### Added

* `CHANGELOG.md` (this file) created for the first time

* `README.md` created for the first time

### Changed

* Package folder structure changed:

    * All modules in `dynsys`: `import` statements revised accordingly
    
    * Documentation in `doc`: this is generated by running `make_docs.py`
    
    * Tests to be in `tests`: this folder is for formal `unittest` routines
    
    * Examples are in `examples`: these are not formal tests, but are examples 
      of the functionality offered by this package
      
    * References (e.g. PDFs of published papers, derivations) should be stored 
      in the `references` folder. These should be linked to via docstrings.