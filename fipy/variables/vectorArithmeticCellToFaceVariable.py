#!/usr/bin/env python

## -*-Pyth-*-
 # ###################################################################
 #  FiPy - Python-based finite volume PDE solver
 # 
 #  FILE: "vectorArithmeticCellToFaceVariable.py"
 #                                    created: 7/26/04 {11:14:05 AM} 
 #                                last update: 7/26/04 {4:00:56 PM} 
 #  Author: Jonathan Guyer
 #  E-mail: guyer@nist.gov
 #  Author: Daniel Wheeler
 #  E-mail: daniel.wheeler@nist.gov
 #    mail: NIST
 #     www: http://ctcms.nist.gov
 #  
 # ========================================================================
 # This software was developed at the National Institute of Standards
 # and Technology by employees of the Federal Government in the course
 # of their official duties.  Pursuant to title 17 Section 105 of the
 # United States Code this software is not subject to copyright
 # protection and is in the public domain.  PFM is an experimental
 # system.  NIST assumes no responsibility whatsoever for its use by
 # other parties, and makes no guarantees, expressed or implied, about
 # its quality, reliability, or any other characteristic.  We would
 # appreciate acknowledgement if the software is used.
 # 
 # This software can be redistributed and/or modified freely
 # provided that any derivative works bear some notice that they are
 # derived from it, and any modified versions bear some notice that
 # they have been modified.
 # ========================================================================
 #  See the file "license.terms" for information on usage and  redistribution
 #  of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 #  
 # ###################################################################
 ##

import Numeric

from fipy.variables.vectorCellToFaceVariable import VectorCellToFaceVariable
from fipy.tools import array
from fipy.tools.inline import inline

class VectorArithmeticCellToFaceVariable(VectorCellToFaceVariable):
    def _calcValuePy(self, alpha, id1, id2):
	cell1 = array.take(self.var,id1)
	cell2 = array.take(self.var,id2)
        self.value = (cell1 - cell2) * alpha[:,Numeric.NewAxis] + cell2
	
    def _calcValueIn(self, alpha, id1, id2):
##        print Numeric.shape(self.value.value)
##        print Numeric.shape(self.var.getNumericValue())
	inline.runInlineLoop2("""
	    double cell2 = var(id2(i),j);
	    val(i,j) = (var(id1(i),j) - cell2) * alpha(i) + cell2;
	""",
	var = self.var.getNumericValue(),
	val = self.value.value, 
	alpha = alpha,
	id1 = id1, id2 = id2,
	ni = self.mesh.getNumberOfFaces(),
        nj = self.mesh.getDim())

	