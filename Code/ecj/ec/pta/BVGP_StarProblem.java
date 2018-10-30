/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ec.pta;
import ec.util.*;
import ec.*;
import ec.app.regression.Benchmarks;
import ec.app.regression.RegressionData;
import ec.gp.*;
import ec.gp.koza.*;
import ec.simple.*;

import java.io.*;
import java.util.*;



/**
 *
 * @author Administrator
 */
public class BVGP_StarProblem extends Benchmarks{
    
	public int[][] dataSet;// = new int[MAXBOOTSAMPLE][trainingInputs.length];
	
	private void bootstrapSampling()
	{	
		dataSet = new int[ec.pta.sfgp.SFProblem.MAXBOOTSAMPLE][trainingInputs.length];

		Random rand = new Random(1000);
		
		for(int i = 0; i < ec.pta.sfgp.SFProblem.MAXBOOTSAMPLE; i++)
		{
			for (int j = 0; j < trainingInputs.length; j++)
				dataSet[i][j] = rand.nextInt(trainingInputs.length);
		}
	}
	
		
	
	public void setup(EvolutionState state, Parameter base)
	{
		super.setup(state, base);
		
		bootstrapSampling();
	}
        
///// Evaluation.  evaluate(...) uses training cases, and describe(...) uses testing cases


    public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum)
    {
        if (!ind.evaluated)  // don't bother reevaluating
        {
            RegressionData input = (RegressionData)(this.input);

            int hits = 0;
            double sum = 0;
            double[] errors = new double[trainingInputs.length];
            
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

			// fabs
            		//   errors[y]=er;

			// CD: Mean Canberra Distance
			//errors[y]=er/(Math.abs(input.x)+Math.abs(trainingOutputs[y]));
        		sum += errors[y];
        		
            }
	    // fabs or CD
            //double fitness = sum/trainingInputs.length;
            
            // RMSE
	    double fitness = sum/trainingInputs.length;
            fitness = Math.sqrt(fitness);
            
            
            double variance = 0;
            double[] bias = new double[ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE];
            double mean = 0;
            for (int i = 0; i < ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE; i++)
            {
            	int index = state.random[0].nextInt(ec.pta.sfgp.SFProblem.MAXBOOTSAMPLE);
            	bias[i] = 0;
            	for(int j = 0; j < trainingInputs.length; j++)
            	{
            		bias[i] += errors[dataSet[index][j]];
            	}
            	//RMSE
            	bias[i] = Math.sqrt(bias[i]/trainingInputs.length);

		// fabs or CD
		//bias[i]=bias[i]/trainingInputs.length;

            //	mean += bias[i];
            }
            
           /* mean = mean/NUMBOOTSAMPLE;
            for(int i = 0; i < bias.length; i++)
            {
            	variance += Math.pow(bias[i] - mean, 2);
            }
            variance = variance / (NUMBOOTSAMPLE- 1);
            */
            sortIncrease(bias);
            for (int i = 0; i < ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER; i++)
                mean += bias[i];
            
            mean = mean/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER;
            for(int i = 0; i < ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER; i++)
            {
            	variance += Math.pow(bias[i] - mean, 2);
            }
            variance = variance / (ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER- 1);
            
            fitness = 0.7 * mean + 0.3 * variance;
            
            // the fitness better be KozaFitness!
            KozaFitness f = ((KozaFitness)ind.fitness);
            f.setStandardizedFitness(state,(float)fitness);
            f.hits = hits;
            
            ind.evaluated = true;
            
            
        }
        
        
        
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
}
