/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package ec.pta.sfgp;
import ec.*;
import ec.app.phishing.Phishing;
import ec.gp.*;
import ec.gp.koza.KozaFitness;

import java.io.*;

import ec.simple.SimpleProblemForm;
import ec.util.*;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.*;


import java.util.List;
import java.util.ArrayList;
import java.util.StringTokenizer;

class Thu_TestPvalue {

    
    public double pValue_KSTest(double []s){
        double p=-1;
        
        // Get a DescriptiveStatistics instance
        DescriptiveStatistics stats = new DescriptiveStatistics();
        // Add the data from the array
        for( int i = 0; i < s.length; i++) {
            stats.addValue(s[i]);
        }
        // Compute some statistics
        double mean = stats.getMean();
        double std = stats.getStandardDeviation();
        double median = stats.getPercentile(50);
       // System.out.println("mean, std"+mean+","+std);
         if (std==0){
            // System.out.println("mean, std"+mean+","+std);
            // for( int i = 0; i < s.length; i++) {
            //    System.out.print(s[i]+" ; ");
            // }
         }   
         else{
            NormalDistribution d=new NormalDistribution(mean, std);
            //Normal d=new Normal(mean,std);
        
            //System.out.println("mean ="+n.getMean()+"  do lech: "+n.getStandardDeviation());
            KolmogorovSmirnovTest t=new KolmogorovSmirnovTest();
            p=t.kolmogorovSmirnovTest(d, s, false);
            //p=DistributionTest.kolmogorov_smirnov_test(s, d);
            // boolean ok=t.kolmogorovSmirnovTest(n, a, 0.05);
        
            //  System.out.println("p value theo thu vien cua minh:"+p);
            //  System.out.println("ok true can reject H0, ok=, :"+String.valueOf(ok));
         }
         
       return p;
    }
 
   
}
class Interval{
    double CI_low;
    double CI_high;
    double mean;
    double median;
    
   
    // luu ket qua du doan cua ca the tot nhat cua 1 job: nhung ton nhieu bo nho qua, hien thoi dung method ben duoi
    public void writeFile(String problem, String alg, int job,  Interval[] arr, String title ){
        try {
            int i;
            String fName; // writer 
            BufferedWriter out;
            if((alg=="sfgp")||(alg=="sf_mssc")){
                //fName="E:/ThucnghiemNCS_2014/ECJ_All/ECJ-Regression_server/out/IntervalPrediction/"+alg+"/Case1/"+problem+"/"+title+"/best.job"+job+".res"; 
                fName="/home/thuongpt/ECJ-Regression/out/IntervalPrediction/" + alg + "/Case1_TTest/"+problem+"/"+title+"/best.job"+job+".res"; 
            } else{
                //fName="E:/ThucnghiemNCS_2014/ECJ_All/ECJ-Regression_server/out/IntervalPrediction/"+alg+"/"+problem+"/"+title+"/best.job"+job+".res"; 
                fName="/home/thuongpt/ECJ-Regression/out/IntervalPrediction/" + alg + "/"+problem+"/"+title+"/best.job"+job+".res"; 
            }
            out= new BufferedWriter(new FileWriter(fName));
            // out.write("Cac khoang du doan cua ca the best: CI-Low "+"\t"+" Mean "+"\t"+" CI-High ");
            // out.newLine();
            for(i=0; i<arr.length;i++){
                out.write(arr[i].CI_low+"\t"+arr[i].mean+"\t"+arr[i].CI_high);
                out.newLine();
            }
            out.close();
        }catch(Exception e ) {
            System.out.println("ERROR: Incorrect data format: "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
    /*
    public void writeFile(GPIndividual t,String problem, String alg, int job,  Interval[] arr, String title ){
        try {
            int i;
            String fName; // writer 
            BufferedWriter out;
            if((alg=="sfgp")||(alg=="sf_mssc")){
                //fName="E:/ThucnghiemNCS_2014/ECJ_All/ECJ-Regression_server/out/IntervalPrediction/"+alg+"/Case1/"+problem+"/"+title+"/best.job"+job+".res"; 
                fName="/home/thuongpt/ECJ-Regression/out/IntervalPrediction/" + alg + "/Case1_TTest/"+problem+"/"+title+"/best.job"+job+".res"; 
            } else{
                //fName="E:/ThucnghiemNCS_2014/ECJ_All/ECJ-Regression_server/out/IntervalPrediction/"+alg+"/"+problem+"/"+title+"/best.job"+job+".res"; 
                fName="/home/thuongpt/ECJ-Regression/out/IntervalPrediction/" + alg + "/"+problem+"/"+title+"/best.job"+job+".res"; 
            }
            out= new BufferedWriter(new FileWriter(fName));
            // out.write("Cac khoang du doan cua ca the best: CI-Low "+"\t"+" Mean "+"\t"+" CI-High ");
            // out.newLine();
            t.predWidth= Math.abs(arr[1].CI_low-arr[1].CI_high);
            t.predAcc=0;
            for(i=0; i<t.valueArr.length;i++){
                if((t.expValueArr[i]>arr[1].CI_low)&&(t.expValueArr[i]<arr[1].CI_high))
                    t.predAcc+=1;
            }
            out.write("row 2: prediction accurate, prediction interval width.");
            out.newLine();
            out.write("Row 3: list of predicted value: seperate by space");
            out.newLine();
            out.write(t.predAcc/t.valueArr.length+"\t"+t.predWidth);
            out.newLine();
            for(i=0; i<t.valueArr.length;i++) // print array prediction values
                out.write(t.valueArr[i]+" ");
            out.close();
        }catch(Exception e ) {
            System.out.println("ERROR: Incorrect data format: "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }*/
}
/* 
 * KozaShortStatistics.java
 * 
 * Created: Fri Nov  5 16:03:44 1999
 * By: Sean Luke
 */

/**
 * A Koza-style statistics generator, intended to be easily parseable with
 * awk or other Unix tools.  Prints fitness information,
 * one generation (or pseudo-generation) per line.
 * If gather-full is true, then timing information, number of nodes
 * and depths of trees, etc. are also given.  No final statistics information
 * is given.
 *
 * <p> Each line represents a single generation.  
 * The first items on a line are always:
 <ul>
 <li> The generation number
 <li> (if gather-full) how long initialization took in milliseconds, or how long the previous generation took to breed to form this generation
 <li> (if gather-full) how many bytes initialization took, or how how many bytes the previous generation took to breed to form this generation.  This utilization is an approximation only, made by the Java system, and does not take into consideration the possibility of garbage collection (which might make the number negative).
 <li> (if gather-full) How long evaluation took in milliseconds this generation
 <li> (if gather-full) how many bytes evaluation took this generation.  This utilization is an approximation only, made by the Java system, and does not take into consideration the possibility of garbage collection (which might make the number negative).
 </ul>

 <p>Then the following items appear, per subpopulation:
 <ul>
 <li> (if gather-full) The average number of nodes used per individual this generation
 <li> (if gather-full) [a|b|c...], representing the average number of nodes used in tree <i>a</i>, <i>b</i>, etc. of individuals this generation
 <li> (if gather-full) The average number of nodes used per individual so far in the run
 <li> (if gather-full) The average depth of any tree per individual this generation
 <li> (if gather-full) [a|b|c...], representing the average depth of tree <i>a</i>, <i>b</i>, etc. of individuals this generation
 <li> (if gather-full) The average depth of any tree per individual so far in the run
 <li> The mean standardized fitness of the subpopulation this generation
 <li> The mean adjusted fitness of the subpopulation this generation
 <li> The mean hits of the subpopulation this generation
 <li> The best standardized fitness of the subpopulation this generation
 <li> The best adjusted fitness of the subpopulation this generation
 <li> The best hits of the subpopulation this generation
 <li> The best standardized fitness of the subpopulation so far in the run
 <li> The best adjusted fitness of the subpopulation so far in the run
 <li> The best hits of the subpopulation so far in the run
 </ul>

 Compressed files will be overridden on restart from checkpoint; uncompressed files will be 
 appended on restart.

 <p><b>Parameters</b><br>
 <table>
 <tr><td valign=top><i>base.</i><tt>gzip</tt><br>
 <font size=-1>boolean</font></td>
 <td valign=top>(whether or not to compress the file (.gz suffix added)</td></tr>
 <tr><td valign=top><i>base.</i><tt>file</tt><br>
 <font size=-1>String (a filename), or nonexistant (signifies stdout)</font></td>
 <td valign=top>(the log for statistics)</td></tr>
 <tr><td valign=top><i>base</i>.<tt>gather-full</tt><br>
 <font size=-1>bool = <tt>true</tt> or <tt>false</tt> (default)</font></td>
 <td valign=top>(should we full statistics on individuals (will run slower, though the slowness is due to off-line processing that won't mess up timings)</td></tr>
 </table>
 * @author Sean Luke
 * @version 1.0 
 */

public class SFStatistics extends Statistics
    {
    public Individual[] getBestSoFar() { return best_of_run; }
    public Individual[] getMedianSoFar(){ return median_of_run;} // PTT @
    /** compress? */
    public static final String P_COMPRESS = "gzip";

    public static final String P_FULL = "gather-full";

    public boolean doFull;

    public Individual[] best_of_run;
    public Individual[] median_of_run;
    
    public long totalNodes[];
    public long totalDepths[];

    // timings
    public long lastTime;
    
    // usage
    public long lastUsage;
    
    /** log file parameter */
    public static final String P_STATISTICS_FILE = "file";

    /** The Statistics' log */
    public int statisticslog;

    public SFStatistics() 
    { /*best_of_run = null;*/ 
        statisticslog = 0; 
     
    /* stdout */ }


    public void setup(final EvolutionState state, final Parameter base)
        {
        super.setup(state,base);  // DO NOT FORGET To call this line
        File statisticsFile = state.parameters.getFile(
            base.push(P_STATISTICS_FILE),null);

        if (statisticsFile!=null) try
                                      {
                                      statisticslog = state.output.addLog(statisticsFile,
                                          !state.parameters.getBoolean(base.push(P_COMPRESS),null,false),
                                          state.parameters.getBoolean(base.push(P_COMPRESS),null,false));
                                      }
            catch (IOException i)
                {
                state.output.fatal("An IOException occurred while trying to create the log " + statisticsFile + ":\n" + i);
                }
             
        doFull = state.parameters.getBoolean(base.push(P_FULL),null,false);

        }


    public void preInitializationStatistics(final EvolutionState state)
        {
        super.preInitializationStatistics(state);

        if (doFull) 
            {
            Runtime r = Runtime.getRuntime();
            lastTime = System.currentTimeMillis();
            lastUsage = r.totalMemory() - r.freeMemory();
            }
        }
    
    public void postInitializationStatistics(final EvolutionState state)
        {
        super.postInitializationStatistics(state);
        // set up our best_of_run array -- can't do this in setup, because
        // we don't know if the number of subpopulations has been determined yet
        best_of_run = new Individual[state.population.subpops.length];
        median_of_run = new Individual[state.population.subpops.length];
        
        // print out our generation number
        state.output.print("0 ", statisticslog);
        //print out numOfSGC, numOfSC
        state.output.print("[ 0 0 0 ] ", statisticslog);

        // gather timings       
        if (doFull)
            {
            totalNodes = new long[state.population.subpops.length];
            for(int x=0;x<totalNodes.length;x++) totalNodes[x] = 0;
            totalDepths = new long[state.population.subpops.length];
            for(int x=0;x<totalDepths.length;x++) totalDepths[x] = 0;
            Runtime r = Runtime.getRuntime();
            long curU =  r.totalMemory() - r.freeMemory();          
            state.output.print("" + (System.currentTimeMillis()-lastTime) + " ",  statisticslog);
            state.output.print("" + (curU-lastUsage) + " ",  statisticslog);            
            }
        }

    public void preBreedingStatistics(final EvolutionState state)
        {
        super.preBreedingStatistics(state);
        state.numOfSGX[state.generation][0] = 0;
        state.numOfSGX[state.generation][1] = 0;
        state.numOfSGX[state.generation][2] = 0;
        
        if (doFull) 
            {
            Runtime r = Runtime.getRuntime();
            lastTime = System.currentTimeMillis();
            lastUsage = r.totalMemory() - r.freeMemory();
            }
        }

    public void postBreedingStatistics(final EvolutionState state) 
        {
        super.postBreedingStatistics(state);
        state.output.print("" + (state.generation + 1) + " ", statisticslog); // 1 because we're putting the breeding info on the same line as the generation it *produces*, and the generation number is increased *after* breeding occurs, and statistics for it

        //print out numOfSGC, numOfSC

        state.output.print("[ " + state.numOfSGX[state.generation][0] + " ", statisticslog);
        state.output.print("" + state.numOfSGX[state.generation][1] + " ", statisticslog);
        state.output.print("" + state.numOfSGX[state.generation][2] + " ]", statisticslog);


        // gather timings
        if (doFull)
            {
            Runtime r = Runtime.getRuntime();
            long curU =  r.totalMemory() - r.freeMemory();          
            state.output.print("" + (System.currentTimeMillis()-lastTime) + " ",  statisticslog);
            state.output.print("" + (curU-lastUsage) + " ",  statisticslog);            
            }
        }

    public void preEvaluationStatistics(final EvolutionState state)
        {
        super.preEvaluationStatistics(state);
        if (doFull) 
            {
            Runtime r = Runtime.getRuntime();
            lastTime = System.currentTimeMillis();
            lastUsage = r.totalMemory() - r.freeMemory();
            }
        }

    /* Prints out the statistics, but does not end with a println --
        this lets overriding methods print additional statistics on the same line 
      
    protected void _postEvaluationStatistics(final EvolutionState state)
        {
        // gather timings
        if (doFull)
            {
            Runtime r = Runtime.getRuntime();
            long curU =  r.totalMemory() - r.freeMemory();          
            state.output.print("" + (System.currentTimeMillis()-lastTime) + " ",  statisticslog);
            state.output.print("" + (curU-lastUsage) + " ",  statisticslog);
            }


        Individual[] best_i = new Individual[state.population.subpops.length];  // quiets compiler complaints
        //PTA - Compute diversity
        int[] num_ind_in_clusters = new int[10]; //10 clusters
        for (int i = 0; i < 10; i++)
        	num_ind_in_clusters[i] = 0;
        int total_individuals = 0;
        //for each subpop
        for(int x=0;x<state.population.subpops.length;x++)
        {
            if (doFull)
                {
                long totNodesPerGen = 0;
                long totDepthPerGen = 0;

                // check to make sure they're the right class
                if ( !(state.population.subpops[x].species instanceof GPSpecies ))
                    state.output.fatal("Subpopulation " + x +
                        " is not of the species form GPSpecies." + 
                        "  Cannot do timing statistics with KozaShortStatistics.");
                
                long[] numNodes = new long[((GPIndividual)(state.population.subpops[x].species.i_prototype)).trees.length];
                long[] numDepth = new long[((GPIndividual)(state.population.subpops[x].species.i_prototype)).trees.length];
                
                for(int y=0;y<state.population.subpops[x].individuals.length;y++)
                    {
                    GPIndividual i = 
                        (GPIndividual)(state.population.subpops[x].individuals[y]);
                    for(int z=0;z<i.trees.length;z++)
                        {
                        numNodes[z] += i.trees[z].child.numNodes(GPNode.NODESEARCH_ALL);
                        numDepth[z] += i.trees[z].child.depth();
                        }
                    }
                
                for(int tr=0;tr<numNodes.length;tr++) totNodesPerGen += numNodes[tr];
                
                totalNodes[x] += totNodesPerGen;


                state.output.print("" + ((double)totNodesPerGen)/state.population.subpops[x].individuals.length + " [",  statisticslog);

                for(int tr=0;tr<numNodes.length;tr++)
                    {
                    if (tr>0) state.output.print("|", statisticslog);
                    state.output.print(""+((double)numNodes[tr])/state.population.subpops[x].individuals.length, statisticslog);
                    }
                state.output.print("] ", statisticslog);

                state.output.print("" + ((double)totalNodes[x])/(state.population.subpops[x].individuals.length * (state.generation + 1)) + " ",
                    statisticslog);

                for(int tr=0;tr<numDepth.length;tr++) totDepthPerGen += numDepth[tr];

                totalDepths[x] += totDepthPerGen;

                state.output.print("" + ((double)totDepthPerGen)/
                        (state.population.subpops[x].individuals.length *
                        numDepth.length) 
                    + " [",  statisticslog);


                for(int tr=0;tr<numDepth.length;tr++)
                    {
                    if (tr>0) state.output.print("|", statisticslog);
                    state.output.print(""+((double)numDepth[tr])/state.population.subpops[x].individuals.length, statisticslog);
                    }
                state.output.print("] ", statisticslog);

                state.output.print("" + ((double)totalDepths[x])/(state.population.subpops[x].individuals.length * (state.generation + 1)) + " ",
                    statisticslog);
                }
            

            
            // fitness information
            float meanStandardized = 0.0f;
            float meanAdjusted = 0.0f;
            long hits = 0;
            
            if (!(state.population.subpops[x].species.f_prototype instanceof KozaFitness))
                state.output.fatal("Subpopulation " + x +
                    " is not of the fitness KozaFitness.  Cannot do timing statistics with KozaStatistics.");

            best_i[x] = null;
            total_individuals += state.population.subpops[x].individuals.length;
            //for each individual in a subpop
            for(int y=0;y<state.population.subpops[x].individuals.length;y++)
            {
                // best individual
                if (best_i[x]==null)
                    best_i[x] = state.population.subpops[x].individuals[y];
                else
                {
                    StochasticFitness f1 = (StochasticFitness)state.population.subpops[x].individuals[y].fitness;
                    StochasticFitness best = (StochasticFitness)best_i[x].fitness;
                    
		    if(f1.betterThan_StatTest(best, state)==1)  // ST
                  //  if(f1.betterThan(best, state)) // PS
                        best_i[x] = state.population.subpops[x].individuals[y];

                }    	
                
            }

            
            //the best of generation i
            StochasticFitness best = (StochasticFitness)best_i[x].fitness;
            
            state.output.print("" + best.standardizedFitness() +
                " " + best.getMean() +
                " " + best.getVariance()+ " ",
                statisticslog);
// -xuat them mang loi of best de ktra outlier
// -xem ca file python
            // find the best so far
            // now test to see if it's the new best_of_run[x]
            if (best_of_run[x]==null)// || best_i[x].fitness.betterThan(best_of_run[x].fitness))
                best_of_run[x] = best_i[x];
            else
            {
            	StochasticFitness best_sofar = (StochasticFitness)best_of_run[x].fitness;
            	if(best.betterThan_StatTest(best_sofar, state)==1)
		//if(best.betterThan(best_sofar, state)) // PS
            		best_of_run[x] = best_i[x];
            }

        	StochasticFitness best_sofar = (StochasticFitness)best_of_run[x].fitness;

            state.output.print("" + best_sofar.standardizedFitness() +
                " " + best_sofar.getMean() +
                " " + best_sofar.getVariance() + " ",
                statisticslog);

            
            if (state.evaluator.p_problem instanceof SimpleProblemForm)
                ((SimpleProblemForm)(state.evaluator.p_problem.clone())).describe(state, best_i[x], x, 0, statisticslog);   

        }
//        //PTA - diversity
//        double diversity = 0.0;
//        for (int i = 0; i < num_ind_in_clusters.length; i++)
//        {
//        	int k = num_ind_in_clusters[i];
//        	if (k > 0)
//        		diversity += (double)k/total_individuals * Math.log((double)k/total_individuals);
//        }
//        diversity = 0 - diversity;
//        state.output.print(" " + diversity +" ", statisticslog);
//        //PTA: Testing
//      if (state.evaluator.p_problem instanceof SimpleProblemForm)
//      ((SimpleProblemForm)(state.evaluator.p_problem.clone())).describe(state, best_of_run[best_of_run.length-1], best_of_run.length-1, 0, statisticslog);   
//
//      best_of_run[best_of_run.length-1].printIndividualForHumans(state, statisticslog);
//        // we're done!
      
        }
*/
    // PTT @

    protected void _postEvaluationStatistics(final EvolutionState state)
        {
        // gather timings
        if (doFull)
            {
            Runtime r = Runtime.getRuntime();
            long curU =  r.totalMemory() - r.freeMemory();          
            state.output.print("" + (System.currentTimeMillis()-lastTime) + " ",  statisticslog);
            state.output.print("" + (curU-lastUsage) + " ",  statisticslog);
            }


        Individual[] best_i = new Individual[state.population.subpops.length];  // quiets compiler complaints
        Individual[] median_i=new Individual[state.population.subpops.length]; // PTT @
        //PTA - Compute diversity
        int[] num_ind_in_clusters = new int[10]; //10 clusters
        for (int i = 0; i < 10; i++)
        	num_ind_in_clusters[i] = 0;
        int total_individuals = 0;
        
        // PTT
        Interval[] arrTAlphaRMSE_noBoot=null;
        Interval[] arrTAlphaMSE_noBoot=null;
        Interval[] arrTAlphaRMSE_Boot=null;
        Interval[] arrTAlphaMSE_Boot=null;
        
        
        Interval obj=new Interval();
        
        //for each subpop
        for(int x=0;x<state.population.subpops.length;x++) // la so subpops: o day subpops = 1
        {
            Individual [] best_tam=new Individual[2]; // store 2 ca the tot nhat (theo bias) cua moi gen
            Individual [] best_run_tam=new Individual[2]; // luu tru 2 thang tot nhat (theo bias) cua moi run
            
            if (doFull)
                {
                long totNodesPerGen = 0;
                long totDepthPerGen = 0;

                // check to make sure they're the right class
                if ( !(state.population.subpops[x].species instanceof GPSpecies ))
                    state.output.fatal("Subpopulation " + x +
                        " is not of the species form GPSpecies." + 
                        "  Cannot do timing statistics with KozaShortStatistics.");
                
                long[] numNodes = new long[((GPIndividual)(state.population.subpops[x].species.i_prototype)).trees.length];
                long[] numDepth = new long[((GPIndividual)(state.population.subpops[x].species.i_prototype)).trees.length];
                
                for(int y=0;y<state.population.subpops[x].individuals.length;y++)
                    {
                    GPIndividual i = 
                        (GPIndividual)(state.population.subpops[x].individuals[y]);
                    for(int z=0;z<i.trees.length;z++)
                        {
                        numNodes[z] += i.trees[z].child.numNodes(GPNode.NODESEARCH_ALL);
                        numDepth[z] += i.trees[z].child.depth();
                        }
                    }
                
                for(int tr=0;tr<numNodes.length;tr++) totNodesPerGen += numNodes[tr];
                
                totalNodes[x] += totNodesPerGen;


                state.output.print("" + ((double)totNodesPerGen)/state.population.subpops[x].individuals.length + " [",  statisticslog);

                for(int tr=0;tr<numNodes.length;tr++)
                    {
                    if (tr>0) state.output.print("|", statisticslog);
                    state.output.print(""+((double)numNodes[tr])/state.population.subpops[x].individuals.length, statisticslog);
                    }
                state.output.print("] ", statisticslog);

                state.output.print("" + ((double)totalNodes[x])/(state.population.subpops[x].individuals.length * (state.generation + 1)) + " ",
                    statisticslog);

                for(int tr=0;tr<numDepth.length;tr++) totDepthPerGen += numDepth[tr];

                totalDepths[x] += totDepthPerGen;

                state.output.print("" + ((double)totDepthPerGen)/
                        (state.population.subpops[x].individuals.length *
                        numDepth.length) 
                    + " [",  statisticslog);


                for(int tr=0;tr<numDepth.length;tr++)
                    {
                    if (tr>0) state.output.print("|", statisticslog);
                    state.output.print(""+((double)numDepth[tr])/state.population.subpops[x].individuals.length, statisticslog);
                    }
                state.output.print("] ", statisticslog);

                state.output.print("" + ((double)totalDepths[x])/(state.population.subpops[x].individuals.length * (state.generation + 1)) + " ",
                    statisticslog);
                }
            

            
             if (!(state.population.subpops[x].species.f_prototype instanceof KozaFitness))
                state.output.fatal("Subpopulation " + x +
                    " is not of the fitness KozaFitness.  Cannot do timing statistics with KozaStatistics.");

            best_i[x] = null;
            
            // PTT @
           // best_tam[0]=null;
           // best_tam[1]=best_i[x];
            
            total_individuals += state.population.subpops[x].individuals.length;
                           
            //for each individual in a subpop
            for(int y=0;y<state.population.subpops[x].individuals.length;y++)
            {
                // best individual
                if (best_i[x]==null){
                    best_i[x] = state.population.subpops[x].individuals[y];
                    
                   // best_tam[0]=best_tam[1];
                   // best_tam[1]=best_i[x];
                }
                else
                {
                   // StochasticFitness f1 = (StochasticFitness)state.population.subpops[x].individuals[y].fitness;
                   // StochasticFitness best = (StochasticFitness)best_i[x].fitness;
                    
                    // PTT @
                    StochasticFitness f1 = (StochasticFitness)state.population.subpops[x].individuals[y].fitness;
                    StochasticFitness best = (StochasticFitness)best_i[x].fitness;
                    // if(f1.betterThan(best,state)){ // PS
                    if(f1.betterThan_StatTest(best,state)==1){ // PTT @ 
                        best_i[x] = state.population.subpops[x].individuals[y];
                     //   best_tam[0]=best_tam[1];
                     //   best_tam[1]=best_i[x];
                    }

                }    	
                
            }
            
            //the best of generation i
            StochasticFitness best = (StochasticFitness)best_i[x].fitness; // best: ca the tot nhat cua mot generation 
            Individual t=best_i[x];
           /* if(best_tam[0]!=null)
                    //if(best.betterThan_Prob((StochasticFitness)best_tam[0].fitness, state)==2){ // dung test mo phong
                     if(!best.betterThan((StochasticFitness)best_tam[0].fitness, state)){ // dung test mo phong
                  
                        best =(StochasticFitness) best_tam[0].fitness;
                        t=best_tam[0];
                    }
            */
            state.output.print("" + best.standardizedFitness() +
                " " + best.getMean() +
                " " + best.getVariance()+ " ",
                statisticslog);
           
             // find the best so far
            // now test to see if it's the new best_of_run[x]
            
            if (best_of_run[x]==null){// || best_i[x].fitness.betterThan(best_of_run[x].fitness))
               // best_run_tam[0]=best_of_run[x];
                best_of_run[x] =t;
            }
            else
            {
            	StochasticFitness best_sofar = (StochasticFitness)best_of_run[x].fitness;
                // if(best.betterThan(best_sofar,state)){ // PS
            	if(best.betterThan_StatTest(best_sofar,state)==1){ // PTT @
                //    best_run_tam[0]=best_of_run[x];
                    best_of_run[x]=t;
                }
            }

        	StochasticFitness best_sofar = (StochasticFitness)best_of_run[x].fitness; // best_sofar: ca the tot nhat cua run
                /*if(best_run_tam[0]!=null)
                        //if(best_sofar.betterThan_Prob((StochasticFitness)best_run_tam[0].fitness, state)==2) // dung test mo phong
                    if(!best_sofar.betterThan((StochasticFitness)best_run_tam[0].fitness, state)) // dung test mo phong        
                        best_sofar =(StochasticFitness) best_run_tam[0].fitness;
                */
            state.output.print("" + best_sofar.standardizedFitness() +
                " " + best_sofar.getMean() +
                " " + best_sofar.getVariance() + " ",
                statisticslog);
            
            if (state.evaluator.p_problem instanceof SimpleProblemForm){
                ((SimpleProblemForm)(state.evaluator.p_problem.clone())).describe(state, t, x, 0, statisticslog); 
            }
        // PTT:  Interval Prediction (OK) - Cach Thg
        //arrTAlphaMSE_noBoot=this.predictionIntervalAlpha_Thg(state, t,0, 0.95,((GPIndividual)t).meanMSE_noBoot ,((GPIndividual)t).varMSE_noBoot,"tAplha_div2", state.freeDegree);
        //obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaMSE_noBoot,"tAlpha_div2_MSE_noBoot"); 
       // arrTAlphaL1_cach2=this.predictionIntervalAlpha_Thg(state, t,0, 0.95,((GPIndividual)t).meanL1_cach2 ,((GPIndividual)t).varL1_cach2,"tAplha", state.freeDegree);
       // obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaL1_cach2,"tAlphaL1_cach2");
       // arrTAlphaRMSE_noBoot=this.predictionIntervalAlpha_Thg(state, t,0, 0.95,((GPIndividual)t).meanRMSE_noBoot ,((GPIndividual)t).varRMSE_noBoot,"tAplha_div2",state.freeDegree);
       // obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaRMSE_noBoot,"tAlpha_div2_RMSE_noBoot"); 
        
        // Cach Thay:
        //arrTAlphaMSE_Boot=this.predictionIntervalAlphaThay(state, t,0, 0.95,((GPIndividual)t).meanMSE_boot ,((GPIndividual)t).varMSE_boot,"tAplha_div2");
        //obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaMSE_Boot,"tAlpha_div2_MSE_boot"); 
       // arrTAlphaL1_cach2=this.predictionIntervalAlphaThay(state, t,0, 0.95,((GPIndividual)t).meanL1_cach2 ,((GPIndividual)t).varL1_cach2,"tAplha");
       // obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaL1_cach2,"tAlphaL1_cach2"); 
       // arrTAlphaRMSE_Boot=this.predictionIntervalAlphaThay(state, t,0, 0.95,((GPIndividual)t).meanRMSE_boot ,((GPIndividual)t).varRMSE_boot,"tAplha_div2");
       // obj.writeFile(state.nameProb,state.nameAlg,state.nameJob, arrTAlphaRMSE_Boot,"tAlpha_div2_RMSE_boot"); 
        
  
        // Thong ke cac ca the co loi bootrap khong tuan theo phan bo chuan su dung KS    
        double pvalue;
        int count=0;
            int couuntNegativePvalue=0;
            List<Integer> pos=new ArrayList<Integer>();
            List<Integer> posN=new ArrayList<Integer>();
            
            for(int y=0;y<state.population.subpops[x].individuals.length;y++){
               // state.output.println("\n Bootstrap errors of individual "+y+": ",statisticslog);
                StochasticFitness tam=(StochasticFitness)state.population.subpops[x].individuals[y].fitness;
                
                double[] b=tam.getErrArr();
                pvalue=new Thu_TestPvalue().pValue_KSTest(b);
                if((pvalue<=0.05)&&(pvalue>=0)) {
                    count++;
                    pos.add(y);
                }
                if(pvalue<0){
                    couuntNegativePvalue++; // so ca the co std bootstrap =0
                    posN.add(y); // mang chua vi tri cac ca the co std bootstrap =0
                }
            }
            state.output.println("\nNum ca the ko normal distribute: ", statisticslog);
            state.output.println(""+count, statisticslog);
            for(int i=0; i< pos.size();i++){
                state.output.print("co: "+couuntNegativePvalue+" ca the co std (bootstrap erro)=0; Indiv:"+pos.get(i)+", Mang loi bootstrap:",statisticslog );
                StochasticFitness tam=(StochasticFitness)state.population.subpops[x].individuals[pos.get(i)].fitness;
                double [] b;
                b=tam.getErrArr();
                for(int j=0;j<b.length; j++){
                   state.output.print(b[j]+" ", statisticslog);
                }
                state.output.print("\n",statisticslog);
            }
          //  state.output.print("\n",statisticslog);
            pvalue=new Thu_TestPvalue().pValue_KSTest(best.getErrArr());
            state.output.println("p value of best: "+pvalue, statisticslog);  
        }
                
        
        
        
            /* PTT @: get errArr of best on generation
            double pvalue;
            int count=0;
            for(int y=0;y<state.population.subpops[x].individuals.length;y++){
               // state.output.println("\n Bootstrap errors of individual "+y+": ",statisticslog);
                StochasticFitness tam=(StochasticFitness)state.population.subpops[x].individuals[y].fitness;
                
                pvalue=new Thu_TestPvalue().pValue_KSTest(tam.getErrArr());
                if(pvalue<0.05) count++;
            }
            state.output.print("\n",statisticslog);
            pvalue=new Thu_TestPvalue().pValue_KSTest(best.getErrArr());
            state.output.println("p value of best: "+pvalue+"; Num ca the ko tuan theo phan bo chuan la: "+count, statisticslog);
            
          }*/
//        //PTA - diversity
//        double diversity = 0.0;
//        for (int i = 0; i < num_ind_in_clusters.length; i++)
//        {
//        	int k = num_ind_in_clusters[i];
//        	if (k > 0)
//        		diversity += (double)k/total_individuals * Math.log((double)k/total_individuals);
//        }
//        diversity = 0 - diversity;
//        state.output.print(" " + diversity +" ", statisticslog);
//        //PTA: Testing
//      if (state.evaluator.p_problem instanceof SimpleProblemForm)
//      ((SimpleProblemForm)(state.evaluator.p_problem.clone())).describe(state, best_of_run[best_of_run.length-1], best_of_run.length-1, 0, statisticslog);   
//
//      best_of_run[best_of_run.length-1].printIndividualForHumans(state, statisticslog);
//        // we're done!
      
        }
    double zAlpha95=1.645; // one side, 95% confidenc
    double zAlpha99=2.33;
    double zAlpha90=1.28;
    
    double zAlpha95_div2=1.96; // two tailed, 95% confidence
    double zAlpha99_div2=2.576;
    double zAlpha90_div2=1.645;
    
    double tAlpha95=2.0452; // (tAlpha_div2):degree of freedom = NumBootstrapAfterRemoveOutlier-1=29
    double tAlpha99=2.4620; // (tAlpha_div2):degree of freedom = NumBootstrapAfterRemoveOutlier-1=29
    double tAlpha90=2.756; // (tAlpha_div2):degree of freedom = NumBootstrapAfterRemoveOutlier-1=29
    
    public Interval[] predictionIntervalAlpha_Thg(EvolutionState state, Individual bestSofar,int threadnum, double level, double mean,double var,String AlphaType,double freeDegree){
        
        GPIndividual t=(GPIndividual)bestSofar;
        
        Interval intervalArr[]=new Interval[t.valueArr.length];
        for(int i=0; i<t.valueArr.length;i++){
           // System.out.println("Test 1:"+((GPIndividual)bestSofar).valueArr[i]);
            intervalArr[i]=new Interval();
            if(level==0.99){
                
            }
            if(level==0.95){
                if(AlphaType=="zAlpha_div2"){
                    // zAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-zAlpha95*Math.sqrt(var/state.numFitcase);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+zAlpha95*Math.sqrt(var/state.numFitcase);
                }else{// ~ tAlpha div 2
                    intervalArr[i].CI_low=t.valueArr[i] -mean-freeDegree*Math.sqrt(var*(1+1/state.numFitcase));///state.numFitcase);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+freeDegree*Math.sqrt(var*(1+1/state.numFitcase));///state.numFitcase);
                   } 
                intervalArr[i].mean=t.valueArr[i]; // phai tinh tren cac run nua, xu ly offline
            }
            if(level==0.9){
            }
        }
        return intervalArr;
    }
    // Cach of Thay
    public Interval[] predictionIntervalAlphaThay(EvolutionState state, Individual bestSofar,int threadnum, double level, double mean,double var,String AlphaType){
       
        GPIndividual t=(GPIndividual)bestSofar;
        
        Interval intervalArr[]=new Interval[t.valueArr.length];
        for(int i=0; i<t.valueArr.length;i++){
           // System.out.println("Test 1:"+((GPIndividual)bestSofar).valueArr[i]);
            intervalArr[i]=new Interval();
            if(level==0.99){
                if(AlphaType=="zAlpha"){
                    // zAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-zAlpha99*Math.sqrt(var/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+zAlpha99*Math.sqrt(var/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                }else{// tAlpha
                    // tAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-tAlpha99*Math.sqrt(var*(1+1/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER)/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+tAlpha99*Math.sqrt(var*(1+1/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER)/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                   }
                intervalArr[i].mean=t.valueArr[i]; // phai tinh tren cac run nua, xu ly offline
            }
            if(level==0.95){
                if(AlphaType=="zAlpha"){
                    // zAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-zAlpha95*Math.sqrt(var);///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+zAlpha95*Math.sqrt(var);///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                }else{
                    // tAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    //intervalArr[i].CI_low=t.valueArr[i] -mean-low*Math.sqrt(var*(1+1/ec.app.regression.Benchmarks.NUMBOOT));///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    //intervalArr[i].CI_high=t.valueArr[i]+mean+up*Math.sqrt(var*(1+1/ec.app.regression.Benchmarks.NUMBOOT));///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_low=t.valueArr[i] -mean-tAlpha95*Math.sqrt(var*(1+1/ec.app.regression.Benchmarks.NUMBOOT));///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+tAlpha95*Math.sqrt(var*(1+1/ec.app.regression.Benchmarks.NUMBOOT));///ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                } 
                intervalArr[i].mean=t.valueArr[i]; // phai tinh tren cac run nua, xu ly offline
            }
            if(level==0.9){
                if(AlphaType=="zAlpha"){
                    // zAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-zAlpha90*Math.sqrt(var/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+zAlpha90*Math.sqrt(var/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                }else{
                    // tAlpha Thay   , mean1, var1 luu mean va var su dung cong thu L1 thay vi dung RMSE
                    intervalArr[i].CI_low=t.valueArr[i] -mean-tAlpha90*Math.sqrt(var*(1+1/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER)/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                    intervalArr[i].CI_high=t.valueArr[i]+mean+tAlpha90*Math.sqrt(var*(1+1/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER)/ec.pta.sfgp.SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
                   } 
                intervalArr[i].mean=t.valueArr[i]; // phai tinh tren cac run nua, xu ly offline
               }
        }
        return intervalArr;
    } 

    
    public void postEvaluationStatistics(final EvolutionState state)
        {
        super.postEvaluationStatistics(state);
        _postEvaluationStatistics(state);
        state.output.println("", statisticslog);
        }

    }


