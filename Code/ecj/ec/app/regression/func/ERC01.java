/*
  Copyright 2012 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package ec.app.regression.func;
import ec.*;
import ec.app.regression.*;
import ec.gp.*;
import ec.util.*;
import java.io.*;


/* 
 * KeijzerERC.java
 * 
 * Created: Wed Nov  3 18:26:37 1999
 * By: Sean Luke

 <p>This ERC appears in the Keijzer function set.  It is defined as a random value drawn from a Gaussian distriution with a mean of 0.0 and a standard deviation of 5.0.

 <p>M. Keijzer. Improving Symbolic Regression with Interval Arithmetic and Linear Scaling. In <i>Proc. EuroGP.</i> 2003.
*/

/**
 * @author Sean Luke
 * @version 1.0 
 */

public class ERC01 extends RegERC
    {
    
    public String name() { return "ERC01"; }

    public void resetNode(final EvolutionState state, final int thread)
        { 
        value = state.random[thread].nextGaussian(); 
        }
    }



