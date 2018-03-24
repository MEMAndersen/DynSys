# DynSys

*Richard Hollamby, COWI UK Bridge, RIHY*

*2018-02-27*

Provides useful classes and functions to faciliate the general dynamic 
analysis of 2nd order structural systems, as characterised by their mass, 
stiffness, damping and constraints.

## Purpose/aim

This package is intended to serve as a structured repository for such content. 
At present all modules are written by Richard Hollamby (COWI UK Bridge, RIHY). 
However the aim is that others should feel able (and are strongly encouraged!) 
to participate by submitting improvements. The only proviso for this is that 
code improvements should be written to be *project-independent*.

The overriding aim is for this package to be reliable, testable, and developed 
in such a way that its performance and capabilities improve with time, due to 
time investment made during projects.

## Applications

Applications include (but are not limited to):

* Time history analysis:

    * Based on full mass, stiffness and damping matrices
    
    * As above, but for systems with multi-point constraints
    
    * For systems represented by a (usually truncated) set of modes
    
    * For systems with *non-proportional* damping
    
* TMD design and analysis

    * Proper account of non-proportional damping, which generally occurs when 
      TMDs are appended to MDOF systems

## Benefits

Benefits of using this package:

* Fully-documented python API (see [Docs link](/docs)), where a rich html API 
  is generated from docstrings in source code using `pdoc` and Markdown syntax.

* All routines written in Python 3, to allow dynamic analysis to be carried 
  out within a more general routine, or used as an `import` in IPython notebook 
  
* As everything is public (in principle) in Python, users familiar with the 
  source code have options:
  
    1. Add new functionality, to extent the capabilities of this package 
    (please consult RIHY to discuss long-term incorporation of improvements)
  
    2. Tweak existing functionality, for a given application
    (this is generally project-specific, so would not be incorporated)
    

