/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/


package ec.select;
import java.util.Random;

import ec.*;

import ec.util.*;
import ec.pta.sfgp.SFProblem;
import ec.pta.sfgp.StochasticFitness;
import ec.simple.SimpleProblemForm;
import ec.steadystate.*;
import ec.pta.sfgp.*;
import static ec.select.SimpleRNG.GetUniform;
import org.apache.commons.math3.distribution.NormalDistribution;

import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.inference.TestUtils;


import org.apache.commons.math3.stat.inference.ChiSquareTest;

/* PTT @ LOP SINH SO NGAU NHIEN THEO CAC DISTRIBUTION  
 * <summary>
    /// SimpleRNG is a simple random number generator based on 
    /// George Marsaglia's MWC (multiply with carry) generator.
    /// Although it is very simple, it passes Marsaglia's DIEHARD
    /// series of random number generator tests.
    /// 
    /// Written by John D. Cook 
    /// http://www.johndcook.com
  * </summary>
  */
  class SimpleRNG
    {
        private static int m_w;
        private static int m_z;

        SimpleRNG()
        {
            // These values are not magical, just the default values Marsaglia used.
            // Any pair of unsigned integers should be fine.
            m_w = 521288629;
            m_z = 362436069;
        }

        // The random generator seed can be set three ways:
        // 1) specifying two non-zero unsigned integers
        // 2) specifying one non-zero unsigned integer and taking a default value for the second
        // 3) setting the seed from the system time

        public static void SetSeed(int u, int v)
        {
            if (u != 0) m_w = u; 
            if (v != 0) m_z = v;
        }

        public static void SetSeed(int u)
        {
            m_w = u;
        }
        /*
        public static void SetSeedFromSystemTime()
        {
            System.DateTime dt = System.DateTime.Now;
            long x = dt.ToFileTime();
            SetSeed((uint)(x >> 16), (uint)(x % 4294967296));
        }
        */ 
        // Produce a uniform random sample from the open interval (0, 1).
        // The method will not return either end point.
        public static double GetUniform()
        {
            // 0 <= u < 2^32
            int u = GetUint();
            // The magic number below is 1/(2^32 + 2).
            // The result is strictly between 0 and 1.
            return (u + 1.0) * 2.328306435454494e-10;
        }

        // This is the heart of the generator.
        // It uses George Marsaglia's MWC algorithm to produce an unsigned integer.
        // See http://www.bobwheeler.com/statistics/Password/MarsagliaPost.txt
        private static int GetUint()
        {
            m_z = 36969 * (m_z & 65535) + (m_z >> 16);
            m_w = 18000 * (m_w & 65535) + (m_w >> 16);
            return (m_z << 16) + m_w;
        }
        
        // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
        public static double GetNormal()
        {
            // Use Box-Muller algorithm
            double u1 = GetUniform();
            double u2 = GetUniform();
            double r = Math.sqrt( -2.0*Math.log(u1) );
            double theta = 2.0*Math.PI*u2;
            return r*Math.sin(theta);
        }
        
        // Get normal (Gaussian) random sample with specified mean and standard deviation
        public static double GetNormal(double mean, double standardDeviation)
        {
            if (standardDeviation <= 0.0)
            {
                String msg = String.format("Shape must be positive. Received {0}.", standardDeviation);
                System.out.println(msg);
            }
            return mean + standardDeviation*GetNormal();
        }
        
        // Get exponential random sample with mean 1
        public static double GetExponential()
        {
            return -Math.log( GetUniform() );
        }

        // Get exponential random sample with specified mean
        public static double GetExponential(double mean)
        {
            if (mean <= 0.0)
            {
                String msg = String.format("Mean must be positive. Received {0}.", mean);
                System.out.println(msg);
            }
            return mean*GetExponential();
        }

        public static double GetGamma(double shape, double scale)
        {
            // Implementation based on "A Simple Method for Generating Gamma Variables"
            // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
            // Vol 26, No 3, September 2000, pages 363-372.

            double d, c, x, xsquared, v, u;

            if (shape >= 1.0)
            {
                d = shape - 1.0/3.0;
                c = 1.0/Math.sqrt(9.0*d);
                for (;;)
                {
                    do
                    {
                        x = GetNormal();
                        v = 1.0 + c*x;
                    }
                    while (v <= 0.0);
                    v = v*v*v;
                    u = GetUniform();
                    xsquared = x*x;
                    if (u < 1.0 -.0331*xsquared*xsquared || Math.log(u) < 0.5*xsquared + d*(1.0 - v + Math.log(v)))
                        return scale*d*v;
                }
            }
            else if (shape <= 0.0)
            {
                String msg = String.format("Shape must be positive. Received {0}.", shape);
                System.out.println(msg+", LOI THUONG THEM VAO");
                return Double.valueOf(msg).doubleValue();
                
            }
            else
            {
                double g = GetGamma(shape+1.0, 1.0);
                double w = GetUniform();
                return scale*g*Math.pow(w, 1.0/shape);
            }
        }

        public static double GetChiSquare(double degreesOfFreedom)
        {
            // A chi squared distribution with n degrees of freedom
            // is a gamma distribution with shape n/2 and scale 2.
            return GetGamma(0.5 * degreesOfFreedom, 2.0);
        }

        public static double GetInverseGamma(double shape, double scale)
        {
            // If X is gamma(shape, scale) then
            // 1/Y is inverse gamma(shape, 1/scale)
            return 1.0 / GetGamma(shape, 1.0 / scale);
        }

        public static double GetWeibull(double shape, double scale)
        {
            if (shape <= 0.0 || scale <= 0.0)
            {
                String msg = String.format("Shape and scale parameters must be positive. Recieved shape {0} and scale{1}.", shape, scale);
                System.out.println(msg);
            }
            return scale * Math.pow(-Math.log(GetUniform()), 1.0 / shape);
        }

        public static double GetCauchy(double median, double scale)
        {
            if (scale <= 0)
            {
                String msg = String.format("Scale must be positive. Received {0}.", scale);
                System.out.println(msg);
            }

            double p = GetUniform();

            // Apply inverse of the Cauchy distribution function to a uniform
            return median + scale*Math.tan(Math.PI*(p - 0.5));
        }

        public static double GetStudentT(double degreesOfFreedom)
        {
            if (degreesOfFreedom <= 0)
            {
                String msg = String.format("Degrees of freedom must be positive. Received {0}.", degreesOfFreedom);
                System.out.println(msg);
            }

            // See Seminumerical Algorithms by Knuth
            double y1 = GetNormal();
            double y2 = GetChiSquare(degreesOfFreedom);
            return y1 / Math.sqrt(y2 / degreesOfFreedom);
        }

        // The Laplace distribution is also known as the double exponential distribution.
        public static double GetLaplace(double mean, double scale)
        {
            double u = GetUniform();
            return (u < 0.5) ?
                mean + scale*Math.log(2.0*u) :
                mean - scale*Math.log(2*(1-u));
        }

        public static double GetLogNormal(double mu, double sigma)
        {
            return Math.exp(GetNormal(mu, sigma));
        }

        public static double GetBeta(double a, double b)
        {
            if (a <= 0.0 || b <= 0.0)
            {
                String msg = String.format("Beta parameters must be positive. Received {0} and {1}.", a, b);
                System.out.println(msg);
            }

            // There are more efficient methods for generating beta samples.
            // However such methods are a little more efficient and much more complicated.
            // For an explanation of why the following method works, see
            // http://www.johndcook.com/distribution_chart.html#gamma_beta

            double u = GetGamma(a, 1.0);
            double v = GetGamma(b, 1.0);
            return u / (u + v);
        }
    }
class StatistisTest extends TTest{
    /**
     * Computes approximate degrees of freedom for 2-sample t-test.
     *
     * @param v1 first sample variance
     * @param v2 second sample variance
     * @param n1 first sample n
     * @param n2 second sample n
     * @return approximate degrees of freedom
     */
    protected double df(double v1, double v2, double n1, double n2) {
        return (((v1 / n1) + (v2 / n2)) * ((v1 / n1) + (v2 / n2))) /
        ((v1 * v1) / (n1 * n1 * (n1 - 1d)) + (v2 * v2) /
                (n2 * n2 * (n2 - 1d)));
    }
}
/* 
 * TournamentSelection.java
 * 
 * Created: Mon Aug 30 19:27:15 1999
 * By: Sean Luke
 */

/**
 * Does a simple tournament selection, limited to the subpopulation it's
 * working in at the time.
 *
 * <p>Tournament selection works like this: first, <i>size</i> individuals
 * are chosen at random from the population.  Then of those individuals,
 * the one with the best fitness is selected.  
 * 
 * <p><i>size</i> can be any floating point value >= 1.0.  If it is a non-
 * integer value <i>x</i> then either a tournament of size ceil(x) is used
 * (with probability x - floor(x)), else a tournament of size floor(x) is used.
 *
 * <p>Common sizes for <i>size</i> include: 2, popular in Genetic Algorithms
 * circles, and 7, popularized in Genetic Programming by John Koza.
 * If the size is 1, then individuals are picked entirely at random.
 *
 * <p>Tournament selection is so simple that it doesn't need to maintain
 * a cache of any form, so many of the SelectionMethod methods just
 * don't do anything at all.
 *

 <p><b>Typical Number of Individuals Produced Per <tt>produce(...)</tt> call</b><br>
 Always 1.

 <p><b>Parameters</b><br>
 <table>
 <tr><td valign=top><i>base.</i><tt>size</tt><br>
 <font size=-1>float &gt;= 1</font></td>
 <td valign=top>(the tournament size)</td></tr>

 <tr><td valign=top><i>base.</i><tt>pick-worst</tt><br>
 <font size=-1> bool = <tt>true</tt> or <tt>false</tt> (default)</font></td>
 <td valign=top>(should we pick the <i>worst</i> individual in the tournament instead of the <i>best</i>?)</td></tr>

 </table>

 <p><b>Default Base</b><br>
 select.tournament

 *
 * @author Sean Luke
 * @version 1.0 
 */

public class ProbabilisticTourSelection extends SelectionMethod implements SteadyStateBSourceForm
    {
    /** default base */
    public static final String P_TOURNAMENT = "tournament";

    public static final String P_PICKWORST = "pick-worst";

    /** size parameter */
    public static final String P_SIZE = "size";

    /** Base size of the tournament; this may change.  */
    int size;

    /** Probablity of picking the size plus one */
    public double probabilityOfPickingSizePlusOne;
    
    /** Do we pick the worst instead of the best? */
    public boolean pickWorst;
    
    static SFProblem problem;// = (SimpleProblemForm)(this.evaluator.p_problem.clone()); // PTT @ cho them static
    

    public Parameter defaultBase()
        {
        return SelectDefaults.base().push(P_TOURNAMENT);
        }
    
    public void setup(final EvolutionState state, final Parameter base)
        {
        super.setup(state,base);
        
        Parameter def = defaultBase();

        double val = state.parameters.getDouble(base.push(P_SIZE),def.push(P_SIZE),1.0);
        if (val < 1.0)
            state.output.fatal("Tournament size must be >= 1.",base.push(P_SIZE),def.push(P_SIZE));
        else if (val == (int) val)  // easy, it's just an integer
            {
            size = (int) val;
            probabilityOfPickingSizePlusOne = 0.0;
            }
        else
            {
            size = (int) Math.floor(val);
            probabilityOfPickingSizePlusOne = val - size;  // for example, if we have 5.4, then the probability of picking *6* is 0.4
            }

        pickWorst = state.parameters.getBoolean(base.push(P_PICKWORST),def.push(P_PICKWORST),false);
        
        
        problem = (SFProblem)(state.evaluator.p_problem.clone());
        
        }

    /** Returns a tournament size to use, at random, based on base size and probability of picking the size plus one. */
    public int getTournamentSizeToUse(MersenneTwisterFast random)
        {
        double p = probabilityOfPickingSizePlusOne;   // pulls us to under 35 bytes
        if (p == 0.0) return size;
        return size + (random.nextBoolean(p) ? 1 : 0);
        }


    /** Produces the index of a (typically uniformly distributed) randomly chosen individual
        to fill the tournament.  <i>number</> is the position of the individual in the tournament.  */
    public int getRandomIndividual(int number, int subpopulation, EvolutionState state, int thread)
        {
        Individual[] oldinds = state.population.subpops[subpopulation].individuals;
        return state.random[thread].nextInt(oldinds.length);
        }

    /* Returns true if *first* is a better (fitter, whatever) individual than *second*. 
    public int betterThan(Individual first, Individual second, 
    		int subpopulation, EvolutionState state, int thread)
        {
        //return first.fitness.betterThan(second.fitness);
    	StochasticFitness f1 = ((StochasticFitness)first.fitness);
    	StochasticFitness f2 = ((StochasticFitness)second.fitness);
    	
    	double subMean = f1.getMean() - f2.getMean();
    	double subStd = Math.sqrt((f1.getVariance() + f2.getVariance())/problem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
    	
    	int count = 0;
    	for(int i = 0; i < problem.NUMTRAIL; i++)
    	{
            /*Gaussian distribution
            double gauss = state.random[0].nextGaussian();
            gauss = gauss * subStd + subMean;
            if (gauss < 0)  
                count += 1;
            
            
            // Student't distribution
            StatistisTest tes=new StatistisTest();
            double t = SimpleRNG.GetStudentT(tes.df(f1.getVariance(), f2.getVariance(), problem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER, problem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER) );
            t = t * subStd + subMean;
            if (t< 0)  
                count += 1;
             
    	}
    	
    	if (count > problem.NUMTRAIL/2) 
            return 1;
    	else 
    		return 2;
    } */
    
// Statistics test
   public static double alpha=0.05;
   public int betterThan(Individual first, Individual second, 
    		int subpopulation, EvolutionState state, int thread)
        { 
        double pValue=0;
        //return first.fitness.betterThan(second.fitness);
    	StochasticFitness f1 = ((StochasticFitness)first.fitness);
    	StochasticFitness f2 = ((StochasticFitness)second.fitness);
    	
    	// Mann Whitney U Test
        //MannWhitneyUTest t=new MannWhitneyUTest();
        //Returns the asymptotic observed significance level, or p-value
        //pValue=t.mannWhitneyUTest(f1.getErrArr(), f2.getErrArr());
        
        // Test with T test
        TTest t =new TTest();
       
        pValue=t.tTest(f1.getErrArr(), f2.getErrArr()); // two tailed: alpha=0.05, one tailed = alpha/2
        
        // confidence level 95% => pValue <0.05
        if(pValue<0.05){
            if(f1.getMean()<f2.getMean())
               return 1; // chose f1
            else
                return 2; // chose f2
        }else
              return 3; // can't reject H0, can choose f1 or f2
            
        
         /*
        double [] t=ztest(f1.getMean(),f2.getMean(),0,f1.getVariance(),f2.getVariance(),f1.getErrArr().length,f2.getErrArr().length);
        if(t[1]<alpha){
            if(f1.getMean()<f2.getMean())
               return 1; // chose f1
            else
                return 2; // chose f2
        }
        else
            return 3; // can't reject H0, can choose f1 or f2
        //return oneTailedZTest(f1,f2, 0.95);
         */
    } 
    // NEW: from (python): TODO return z score and pvalue. X1,X2 : 2 samples, mudiff =0, n1, n2: size of sample
    public static double[] ztest(double meanX1, double meanX2, int mudiff, double var1, double var2, int n1, int n2){
        double tam[]=new double[2];
        NormalDistribution d=new NormalDistribution();
        double pooledSE = Math.sqrt(var1/n1 + var2/n2);
        double z = ((meanX1 - meanX2) - mudiff)/pooledSE; // mufiff=0=mean cua phan bo N(0,1)
        // pval = 2*(1 - norm.cdf(abs(z)))
        double pval = 2*(1 - d.cumulativeProbability(Math.abs(z))); // d: standard normal distribution
        tam[0]=z;
        tam[1]=pval;
        return tam;
        
        //z, p = twoSampZ(28, 33, 0, 14.1, 9.5, 75, 50)
        //print (z, p)
    }
            
    public int produce(final int subpopulation,
        final EvolutionState state,
        final int thread)
        {
        // pick size random individuals, then pick the best.
        
        Individual[] oldinds = state.population.subpops[subpopulation].individuals;
        int best = getRandomIndividual(0, subpopulation, state, thread);
        
        int s = getTournamentSizeToUse(state.random[thread]);
                
        if (pickWorst)
            for (int x=1;x<s;x++){
                int j = getRandomIndividual(x, subpopulation, state, thread);
                //if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)==1){// PS
                if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)==1){  // ST
                    best = j; // chose j
                }else{
                    if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)==3) 
                        best=Class_Random.choseRandom(j,best); // chose random
                    else{
                        //keep best
                    }
                }
            }
        else
            for (int x=1;x<s;x++){
                int j = getRandomIndividual(x, subpopulation, state, thread);
                //if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)){ // PS
                  if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)==1){  // j is better than best
                    best = j;
                  }else{
                    if (betterThan(oldinds[j], oldinds[best], subpopulation, state, thread)==3) 
                        best=Class_Random.choseRandom(j,best);
                }
            }
            
        return best;
        }

    // included for SteadyState
    public void individualReplaced(final SteadyStateEvolutionState state,
        final int subpopulation,
        final int thread,
        final int individual)
        { return; }
    
    public void sourcesAreProperForm(final SteadyStateEvolutionState state)
        { return; }
    
    }
interface Const {
	public static final int		MAXSTRING	= 60000;				// max of a tring (for _sbuffer)
	public static final byte	TRUE			= 1;
	public static final byte	FALSE			= 0;
	public static final int		MAXNAME		= 6;						// max+1 length of name of a symbol
	public static final int		MAXDEPTH		= 15;					// max size of chromosome
	public static final int		MAXTOUR		= 10;					// max tournement size
	public static final int		MAXATEMPT	= 100;					// max atempt for choosing a site of a chromosome
	public static final int		MAXFUNCTION	= 20;
	public static final int		MAXTERMINAL	= 20;
	public static final int		MAXNODE		= 3000;
	public static final double	VOIDVALUE	= -1523612789.21342;
	public static final int		NUMFITCASE	= 100;//300;//50;//20;//50;//300;//100;
	public static final int		NUMFITTEST	= 220;//2701;//90601;//1089;//361201;//1089;//961;//2701;//220;

	public static final int		NUMVAR	=1;//3;//2;//3;//1;
	
	public static final double	INFPLUS		=  Math.exp(700);
	public static final double	INFMINUS		= -Math.exp(700);
	public static final double	HUGE_VAL		= Math.exp(700);
        public static final int         N               =1;//31;//1;//31;//1;// 51;//31;//10; // number times drop randomly to compare 2 individula
        public static final int         B               = 50;//35;//500;//10;//;30;//50;//500;//1000;//10000;//100000; // draw B bootstrap data sets
        public static final int         MAXBOOTSAMPLE   = 1500;// Max num bootstrap sample sets
        public static final int         B1      =30; // so mau can lay de hyphothesis test
}

class Class_Random {

	static long		idum	= (long) -12345;
	static long		IA		= (long) 16807;
	static long		IM		= 2147483647;
	static double	AM		= (1.0 / IM);
	static long		IQ		= 127773;
	static long		IR		= (long) 2836;
	static long		NTAB	= 32;
	static double	NDIV	= (1.0 + (IM - 1.0) / (double) NTAB);
	static double	EPS	= 1.2e-7;
	static double	RNMX	= (1.0 - EPS);
	static long		iy		= 0;
	static long[]	iv		= new long[(int) NTAB];//???
	
	public Class_Random(){}
	
	//void Set_Seed(int x){
		//idum = (long) -x;
	//}

	void Set_Seed(long x){
		idum = -x;
	}
	
	static double Next_Double(){
		int j;
		long k;
		double temp;
		//System.out.print("idum3="+getidum());
		if(idum <= 0 || iy == 0)//??? 
		{
			if(-idum < 1) idum = 1;
			else idum = -idum;
			for(j = (int) NTAB + 7; j >= 0; j--) {
				k = (long) (idum / (double) IQ);
				idum = IA * (idum - k * IQ) - IR * k;
				if(idum < 0) idum += IM;
				if(j < NTAB) iv[j] = idum;
			}
			iy = iv[0];
		}
		k = (long) (idum / (double) IQ);
		idum = IA * (idum - k * IQ) - IR * k;
		if(idum < 0) idum += IM;
		j = (int) (iy / NDIV);
		iy = iv[j];
		iv[j] = idum;
		if((temp = AM * iy) > RNMX) return RNMX;
		else return temp;
	}
        static byte Flip(double prob){
		double temp = Next_Double();
		if (temp<=prob) return 1;
		else
			return 0;		
	}

// return a random integer between lower and upper
	static int IRandom(int lower, int upper){
		int temp;
		temp = lower + (int) (Next_Double() * (upper - lower + 1));
		return temp;
	}
        static int choseRandom(int a, int b){
            byte t;
            t=Flip(0.5);
            if (t==1){
                return a;
            }else{
                return b;
            }
            
            
        }
        // return a random double between lower and upper
        static double DRandom(double lower, double upper){
		double temp;
		temp = lower + (Next_Double() * (upper - lower+0.01));
		return temp;
	}
}

