"""
Use pdoc to make html documentation for modules
"""

from __init__ import __version__ as currentVersion

import pdoc
import markdown
import os

def make_html(moduleName_full):
    
    print("Creating docs for `{0}` module".format(moduleName_full))
    
    h = pdoc.html(moduleName_full)
    h = "Version: {0}".format(currentVersion) + "\n" + h
    
    moduleName = moduleName_full.split('.')[-1]
    
    f = open("../docs/{0}.html".format(moduleName), 'w')
    
    f.write(h)
    f.close()
    
def markdown_to_html(fName_input):
    
    print("Rendering markdown text from {0}".format(fName_input))
    
    fName_output = "../docs/{0}.html".format(os.path.splitext(fName_input)[0])

    markdown.markdownFromFile(input=("../" + fName_input),
                              output=fName_output)
    
   #%% 
print("Creating documentation for version {0}".format(currentVersion))
   
# List modulues to be documented here
make_html("dynsys")
make_html("modalsys")
make_html("msd_chain")
make_html("hanging_1d_chain")
make_html("tstep")
make_html("tstep_results")
make_html("eig_results")
make_html("freq_response_results")
make_html("dyn_analysis")
make_html("loading")
make_html("ped_dyn")
make_html("damper")
make_html("mesh")
make_html("nodle")
make_html("wind_section")
make_html("wind_env")
make_html("wind_response")

#%%
# List markdown files in root folder to be rendered
markdown_to_html("README.md")
markdown_to_html("CHANGELOG.md")




