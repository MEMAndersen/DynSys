"""
Use pdoc to make html documentation for test modules
"""

import pdoc

def make_html(moduleName_full):
    
    h = pdoc.html(moduleName_full)
    
    moduleName = moduleName_full.split('.')[-1]
    
    f = open("docs/{0}.html".format(moduleName), 'w')
    
    f.write(h)
    f.close()
    
# List modulues to be documented here
make_html("warburton_TMD")