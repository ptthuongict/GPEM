package ec.pta.sfgp;

import ec.util.*;
import ec.*;
import ec.app.regression.Benchmarks;
import ec.app.regression.RegressionData;
import ec.gp.*;
import ec.gp.koza.*;
import ec.simple.*;

import java.io.*;
import java.util.*;

/* 
 * Benchmarks.java
 * U    
 * Created: Thu Jul 14 10:35:11 EDT 2011
 * By: Sean Luke
 *
 * This is an expansion of the Regression.java file to provide a first cut at a standardized
 * regression benchmark for Genetic Programming.  The package provides both training and
 * testing data and functions for the following problems:
 
 <ol>
 <li>John Koza's three  problems (quartic, quintic, and sextic, drawn from his GP-1 and GP-2 books.  These are known as <b>koza-1</b> through <b>koza-3</b>
 <li>Twelve problems drawn from "Semantically-based Crossover in Genetic Programming: Application to Real-valued Symbolic Regression" (GPEM 2011),
 by Nguyen Quang Uy, Nguyen Xuan Hoai, Michael , R.I. McKay, Edgar Galv .  These are known as
 <b>nguyen-1</b> through <b>nguyen-10</b>
 <li>Fifteen problems drawn from "Accuracy in Symbolic Regression" (GPEM 2011), by Michael Korns.  These are known as <b>KORNS1</b> through <b>KORNS15</b>
 <li>Fifteen problems drawn from "Improving Symbolic Regression with Interval Arithmetic and Linear Scaling" (EuroGP 2003), by Maarten Keijzer.  These are known as <b>keijzer-1</b> through <b>keijzer-15</b>
 <li>Fifteen problems drawn from "Order of Nonlinearity as a Complexity Measure for Models Generated by Symbolic Regression via Pareto Genetic Programming" (IEEE TransEC 13:2), by Ekaterina J. Vladislavleva, Guido F. Smits, and Dick den Hertog.  These are known as <b>vladislavleva-1</b> through <b>vladislavleva-8</b>
 <li>Two problems drawn from "Evolutionary Consequences of Coevolving Targets" (Evolutionary Computation 5(4)) by Ludo Pagie and Paulien Hogeweg, 1997.
 <li>You can also provide your own data sets via a text file.

 <p>These problems differ in a variety of ways.  Some have both training and testing data points.  Some have one variable, others up to five variables.  Some build their random points based on random samples, other based on a grid of samples distributed through the space.  Different problems also use different function sets.
 
 <p>The functions below are all described precisely in the paper "Genetic Programming Needs Better Benchmarks" (GECCO 2012) by James McDermmott, David R. white, Sean Luke, Luca Manzoni, Mauro Castelli, Leonardo Vanneschi, Wojciech Jaskowski, Krzysztof Krawiec, Robin Harper, Kenneth De Jong, and Una-May O'Reilly.  The descriptions, shown in Tables 2 and 3, explain the function set, number of variables, objective function, training set, testing set, and source of each problem.
 
 <p>In addition we include one more function: PAGIE2, a 3-variable version of PAGIE1. We describe PAGIE2 as follows:
 
 <ul>
 <li>Variables: 3
 li>Function: (1 / (1 + x^(-4)) + 1 / (1 + y^(-4)) + 1 / (1 + z^(-4)))
 <li>Testing Set: Grid points from -5 to 5 respectively in each dimension, spaced out in intervals of 0.4.
 <li>Training Set: none
 <li>Function Set: standard Koza with three terminals (X1, X2, X3).
 </ul>

 *
 *
 */

/**
 * Benchmarks by various people in the literature.
 *  
 *  
 *  Stochastic Fitness GP
 */
public class SFProblem extends Benchmarks
    {
	
	/**
     * Stochastic Fitness
     */
    public static final int MAXBOOTSAMPLE = 1500;
	public static final int NUMBOOTSAMPLE = 30;//70;//50; //B
        public static final int NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER=30;//50;
	public final int NUMTRAIL = 1;
	public int[][] dataSet;// = new int[MAXBOOTSAMPLE][trainingInputs.length];
	
	private void bootstrapSampling()
	{		
		dataSet = new int[MAXBOOTSAMPLE][trainingInputs.length];

		Random rand = new Random(1000);
		
		for(int i = 0; i < MAXBOOTSAMPLE; i++)
		{
			for (int j = 0; j < trainingInputs.length; j++)
				dataSet[i][j] = rand.nextInt(trainingInputs.length);
		}
	}
	
		
	
	public void setup(EvolutionState state, Parameter base)
	{
		super.setup(state, base);
		bootstrapSampling();
                state.numFitcase=trainingInputs.length;
	}
        // sort increase
        public void sortIncrease(double[] bias){
            int i,j,k;
            double tam;
            for(i=0;i<bias.length-1;i++){
                k=i;
                for(j=k+1;j<bias.length;j++)
                    if(bias[j]<bias[k])
                        k=j;
                if(k!=i){
                    tam=bias[i];
                    bias[i]=bias[k];
                    bias[k]=tam;
                }
            }
        }
///// Evaluation.  evaluate(...) uses training cases, and describe(...) uses testing cases


    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum)
    {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            RegressionData input = (RegressionData)(this.input);

            int hits = 0;
            double sum = 0.0;
            double[] errors = new double[trainingInputs.length];
            state.inputTrainThg=input;
            for (int y=0;y<trainingInputs.length;y++)
            {
            	double error = 0;
            	
        		currentValue = trainingInputs[y];
        		((GPIndividual)ind).trees[0].child.eval(
        				state,threadnum,input,stack,((GPIndividual)ind),this);
            
        		//store semantic of y-th element
        		((GPIndividual)ind).trees[0].semanticTraining[y] = input.x;
                        	            		
        		double er = error(input.x, trainingOutputs[y]);
        		
        		// RMSE
        		errors[y] = er * er;
			
			//fabs
			//errors[y] = er; 

			// CD: Mean Canberra Distance
			//errors[y]=er/(Math.abs(input.x)+Math.abs(trainingOutputs[y]));
        		
        		sum += errors[y];
            }
            
           float fitness = (float)sum/trainingInputs.length;
	    // RMSE
            fitness = (float)Math.sqrt(fitness);
                
            double mean = 0;
            double variance = 0;
            double[] bias = new double[NUMBOOTSAMPLE];
            
            for (int i = 0; i < NUMBOOTSAMPLE; i++)
            {
            	int index = state.random[0].nextInt(MAXBOOTSAMPLE);
            	bias[i] = 0;
            	for(int j = 0; j < trainingInputs.length; j++)
            	{
            		bias[i] += errors[dataSet[index][j]];
            	}
            	//RMSE
            	bias[i] = Math.sqrt(bias[i]/trainingInputs.length);

		// fabs or CD
		//bias[i] = bias[i]/trainingInputs.length;
               
               //  mean += bias[i];
            }
           /* mean = mean/NUMBOOTSAMPLE;
            for(int i = 0; i < NUMBOOTSAMPLE; i++)
            {
            	variance += Math.pow(bias[i] - mean, 2);
            }
            variance = variance / (NUMBOOTSAMPLE- 1);
            */ 
            
            sortIncrease(bias);
            for (int i = 0; i < NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER; i++)
                mean += bias[i];
            
            mean = mean/NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER;
            for(int i = 0; i < NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER; i++)
            {
            	variance += Math.pow(bias[i] - mean, 2);
            }
            variance = variance / (NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER- 1);
            
            
            // the fitness better be KozaFitness!
            StochasticFitness f = ((StochasticFitness)ind.fitness);
	 
            f.setStandardizedFitness(state, fitness); //no need
            f.setErrArr(bias,NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
            f.setMean(mean);
            f.setVariance(variance);
           
            f.setStandardizedFitness(state, (float)f.getMean()); // Thuong @
            ind.evaluated = true;
            
            
        }
        
        
        
    }

        
    }
