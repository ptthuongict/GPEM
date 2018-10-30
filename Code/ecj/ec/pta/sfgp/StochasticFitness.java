/*
  Copyright 2006 by Sean Luke
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/ 


package ec.pta.sfgp;
import ec.gp.koza.GPKozaDefaults;
import ec.gp.koza.KozaFitness;
import ec.util.*;
import ec.*;
import static ec.pta.sfgp.SimpleRNG.GetUniform;
import java.io.*;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.StatisticalSummary;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import org.apache.commons.math3.util.FastMath;

class TTest {
    /**
     * Computes a paired, 2-sample t-statistic based on the data in the input
     * arrays.  The t-statistic returned is equivalent to what would be returned by
     * computing the one-sample t-statistic {@link #t(double, double[])}, with
     * <code>mu = 0</code> and the sample array consisting of the (signed)
     * differences between corresponding entries in <code>sample1</code> and
     * <code>sample2.</code>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The input arrays must have the same length and their common length
     * must be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return t statistic
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NoDataException if the arrays are empty
     * @throws DimensionMismatchException if the length of the arrays is not equal
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     */
    public double pairedT(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NoDataException,
        DimensionMismatchException, NumberIsTooSmallException {

        checkSampleData(sample1);
        checkSampleData(sample2);
        double meanDifference = StatUtils.meanDifference(sample1, sample2);
        return t(meanDifference, 0,
                 StatUtils.varianceDifference(sample1, sample2, meanDifference),
                 sample1.length);

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i> p-value</i>, associated with a paired, two-sample, two-tailed t-test
     * based on the data in the input arrays.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the mean of the paired
     * differences is 0 in favor of the two-sided alternative that the mean paired
     * difference is not equal to 0. For a one-sided test, divide the returned
     * value by 2.</p>
     * <p>
     * This test is equivalent to a one-sample t-test computed using
     * {@link #tTest(double, double[])} with <code>mu = 0</code> and the sample
     * array consisting of the signed differences between corresponding elements of
     * <code>sample1</code> and <code>sample2.</code></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the p-value depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The input array lengths must be the same and their common length must
     * be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return p-value for t-test
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NoDataException if the arrays are empty
     * @throws DimensionMismatchException if the length of the arrays is not equal
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double pairedTTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NoDataException, DimensionMismatchException,
        NumberIsTooSmallException, MaxCountExceededException {

        double meanDifference = StatUtils.meanDifference(sample1, sample2);
        return tTest(meanDifference, 0,
                StatUtils.varianceDifference(sample1, sample2, meanDifference),
                sample1.length);

    }

    /**
     * Performs a paired t-test evaluating the null hypothesis that the
     * mean of the paired differences between <code>sample1</code> and
     * <code>sample2</code> is 0 in favor of the two-sided alternative that the
     * mean paired difference is not equal to 0, with significance level
     * <code>alpha</code>.
     * <p>
     * Returns <code>true</code> iff the null hypothesis can be rejected with
     * confidence <code>1 - alpha</code>.  To perform a 1-sided test, use
     * <code>alpha * 2</code></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The input array lengths must be the same and their common length
     * must be at least 2.
     * </li>
     * <li> <code> 0 &lt; alpha &lt; 0.5 </code>
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with
     * confidence 1 - alpha
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NoDataException if the arrays are empty
     * @throws DimensionMismatchException if the length of the arrays is not equal
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public boolean pairedTTest(final double[] sample1, final double[] sample2,
                               final double alpha)
        throws NullArgumentException, NoDataException, DimensionMismatchException,
        NumberIsTooSmallException, OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return pairedTTest(sample1, sample2) < alpha;

    }

    /**
     * Computes a <a href="http://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm#formula">
     * t statistic </a> given observed values and a comparison constant.
     * <p>
     * This statistic can be used to perform a one sample t-test for the mean.
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array length must be at least 2.
     * </li></ul></p>
     *
     * @param mu comparison constant
     * @param observed array of values
     * @return t statistic
     * @throws NullArgumentException if <code>observed</code> is <code>null</code>
     * @throws NumberIsTooSmallException if the length of <code>observed</code> is &lt; 2
     */
    public double t(final double mu, final double[] observed)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(observed);
        // No try-catch or advertised exception because args have just been checked
        return t(StatUtils.mean(observed), mu, StatUtils.variance(observed),
                observed.length);

    }

    /**
     * Computes a <a href="http://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm#formula">
     * t statistic </a> to use in comparing the mean of the dataset described by
     * <code>sampleStats</code> to <code>mu</code>.
     * <p>
     * This statistic can be used to perform a one sample t-test for the mean.
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li><code>observed.getN() &ge; 2</code>.
     * </li></ul></p>
     *
     * @param mu comparison constant
     * @param sampleStats DescriptiveStatistics holding sample summary statitstics
     * @return t statistic
     * @throws NullArgumentException if <code>sampleStats</code> is <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     */
    public double t(final double mu, final StatisticalSummary sampleStats)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(sampleStats);
        return t(sampleStats.getMean(), mu, sampleStats.getVariance(),
                 sampleStats.getN());

    }

    /**
     * Computes a 2-sample t statistic,  under the hypothesis of equal
     * subpopulation variances.  To compute a t-statistic without the
     * equal variances hypothesis, use {@link #t(double[], double[])}.
     * <p>
     * This statistic can be used to perform a (homoscedastic) two-sample
     * t-test to compare sample means.</p>
     * <p>
     * The t-statistic is</p>
     * <p>
     * &nbsp;&nbsp;<code>  t = (m1 - m2) / (sqrt(1/n1 +1/n2) sqrt(var))</code>
     * </p><p>
     * where <strong><code>n1</code></strong> is the size of first sample;
     * <strong><code> n2</code></strong> is the size of second sample;
     * <strong><code> m1</code></strong> is the mean of first sample;
     * <strong><code> m2</code></strong> is the mean of second sample</li>
     * </ul>
     * and <strong><code>var</code></strong> is the pooled variance estimate:
     * </p><p>
     * <code>var = sqrt(((n1 - 1)var1 + (n2 - 1)var2) / ((n1-1) + (n2-1)))</code>
     * </p><p>
     * with <strong><code>var1</code></strong> the variance of the first sample and
     * <strong><code>var2</code></strong> the variance of the second sample.
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return t statistic
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     */
    public double homoscedasticT(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(sample1);
        checkSampleData(sample2);
        // No try-catch or advertised exception because args have just been checked
        return homoscedasticT(StatUtils.mean(sample1), StatUtils.mean(sample2),
                              StatUtils.variance(sample1), StatUtils.variance(sample2),
                              sample1.length, sample2.length);

    }

    /**
     * Computes a 2-sample t statistic, without the hypothesis of equal
     * subpopulation variances.  To compute a t-statistic assuming equal
     * variances, use {@link #homoscedasticT(double[], double[])}.
     * <p>
     * This statistic can be used to perform a two-sample t-test to compare
     * sample means.</p>
     * <p>
     * The t-statistic is</p>
     * <p>
     * &nbsp;&nbsp; <code>  t = (m1 - m2) / sqrt(var1/n1 + var2/n2)</code>
     * </p><p>
     *  where <strong><code>n1</code></strong> is the size of the first sample
     * <strong><code> n2</code></strong> is the size of the second sample;
     * <strong><code> m1</code></strong> is the mean of the first sample;
     * <strong><code> m2</code></strong> is the mean of the second sample;
     * <strong><code> var1</code></strong> is the variance of the first sample;
     * <strong><code> var2</code></strong> is the variance of the second sample;
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return t statistic
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     */
    public double t(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(sample1);
        checkSampleData(sample2);
        // No try-catch or advertised exception because args have just been checked
        return t(StatUtils.mean(sample1), StatUtils.mean(sample2),
                 StatUtils.variance(sample1), StatUtils.variance(sample2),
                 sample1.length, sample2.length);

    }

    /**
     * Computes a 2-sample t statistic </a>, comparing the means of the datasets
     * described by two {@link StatisticalSummary} instances, without the
     * assumption of equal subpopulation variances.  Use
     * {@link #homoscedasticT(StatisticalSummary, StatisticalSummary)} to
     * compute a t-statistic under the equal variances assumption.
     * <p>
     * This statistic can be used to perform a two-sample t-test to compare
     * sample means.</p>
     * <p>
      * The returned  t-statistic is</p>
     * <p>
     * &nbsp;&nbsp; <code>  t = (m1 - m2) / sqrt(var1/n1 + var2/n2)</code>
     * </p><p>
     * where <strong><code>n1</code></strong> is the size of the first sample;
     * <strong><code> n2</code></strong> is the size of the second sample;
     * <strong><code> m1</code></strong> is the mean of the first sample;
     * <strong><code> m2</code></strong> is the mean of the second sample
     * <strong><code> var1</code></strong> is the variance of the first sample;
     * <strong><code> var2</code></strong> is the variance of the second sample
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The datasets described by the two Univariates must each contain
     * at least 2 observations.
     * </li></ul></p>
     *
     * @param sampleStats1 StatisticalSummary describing data from the first sample
     * @param sampleStats2 StatisticalSummary describing data from the second sample
     * @return t statistic
     * @throws NullArgumentException if the sample statistics are <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     */
    public double t(final StatisticalSummary sampleStats1,
                    final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(sampleStats1);
        checkSampleData(sampleStats2);
        return t(sampleStats1.getMean(), sampleStats2.getMean(),
                 sampleStats1.getVariance(), sampleStats2.getVariance(),
                 sampleStats1.getN(), sampleStats2.getN());

    }

    /**
     * Computes a 2-sample t statistic, comparing the means of the datasets
     * described by two {@link StatisticalSummary} instances, under the
     * assumption of equal subpopulation variances.  To compute a t-statistic
     * without the equal variances assumption, use
     * {@link #t(StatisticalSummary, StatisticalSummary)}.
     * <p>
     * This statistic can be used to perform a (homoscedastic) two-sample
     * t-test to compare sample means.</p>
     * <p>
     * The t-statistic returned is</p>
     * <p>
     * &nbsp;&nbsp;<code>  t = (m1 - m2) / (sqrt(1/n1 +1/n2) sqrt(var))</code>
     * </p><p>
     * where <strong><code>n1</code></strong> is the size of first sample;
     * <strong><code> n2</code></strong> is the size of second sample;
     * <strong><code> m1</code></strong> is the mean of first sample;
     * <strong><code> m2</code></strong> is the mean of second sample
     * and <strong><code>var</code></strong> is the pooled variance estimate:
     * </p><p>
     * <code>var = sqrt(((n1 - 1)var1 + (n2 - 1)var2) / ((n1-1) + (n2-1)))</code>
     * </p><p>
     * with <strong><code>var1</code></strong> the variance of the first sample and
     * <strong><code>var2</code></strong> the variance of the second sample.
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The datasets described by the two Univariates must each contain
     * at least 2 observations.
     * </li></ul></p>
     *
     * @param sampleStats1 StatisticalSummary describing data from the first sample
     * @param sampleStats2 StatisticalSummary describing data from the second sample
     * @return t statistic
     * @throws NullArgumentException if the sample statistics are <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     */
    public double homoscedasticT(final StatisticalSummary sampleStats1,
                                 final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException {

        checkSampleData(sampleStats1);
        checkSampleData(sampleStats2);
        return homoscedasticT(sampleStats1.getMean(), sampleStats2.getMean(),
                              sampleStats1.getVariance(), sampleStats2.getVariance(),
                              sampleStats1.getN(), sampleStats2.getN());

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a one-sample, two-tailed t-test
     * comparing the mean of the input array with the constant <code>mu</code>.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the mean equals
     * <code>mu</code> in favor of the two-sided alternative that the mean
     * is different from <code>mu</code>. For a one-sided test, divide the
     * returned value by 2.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">here</a>
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array length must be at least 2.
     * </li></ul></p>
     *
     * @param mu constant value to compare sample mean against
     * @param sample array of sample data values
     * @return p-value
     * @throws NullArgumentException if the sample array is <code>null</code>
     * @throws NumberIsTooSmallException if the length of the array is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double tTest(final double mu, final double[] sample)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sample);
        // No try-catch or advertised exception because args have just been checked
        return tTest(StatUtils.mean(sample), mu, StatUtils.variance(sample),
                     sample.length);

    }

    /**
     * Performs a <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm">
     * two-sided t-test</a> evaluating the null hypothesis that the mean of the population from
     * which <code>sample</code> is drawn equals <code>mu</code>.
     * <p>
     * Returns <code>true</code> iff the null hypothesis can be
     * rejected with confidence <code>1 - alpha</code>.  To
     * perform a 1-sided test, use <code>alpha * 2</code></p>
     * <p>
     * <strong>Examples:</strong><br><ol>
     * <li>To test the (2-sided) hypothesis <code>sample mean = mu </code> at
     * the 95% level, use <br><code>tTest(mu, sample, 0.05) </code>
     * </li>
     * <li>To test the (one-sided) hypothesis <code> sample mean < mu </code>
     * at the 99% level, first verify that the measured sample mean is less
     * than <code>mu</code> and then use
     * <br><code>tTest(mu, sample, 0.02) </code>
     * </li></ol></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the one-sample
     * parametric t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/sg_glos.html#one-sample">here</a>
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array length must be at least 2.
     * </li></ul></p>
     *
     * @param mu constant value to compare sample mean against
     * @param sample array of sample data values
     * @param alpha significance level of the test
     * @return p-value
     * @throws NullArgumentException if the sample array is <code>null</code>
     * @throws NumberIsTooSmallException if the length of the array is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error computing the p-value
     */
    public boolean tTest(final double mu, final double[] sample, final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return tTest(mu, sample) < alpha;

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a one-sample, two-tailed t-test
     * comparing the mean of the dataset described by <code>sampleStats</code>
     * with the constant <code>mu</code>.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the mean equals
     * <code>mu</code> in favor of the two-sided alternative that the mean
     * is different from <code>mu</code>. For a one-sided test, divide the
     * returned value by 2.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The sample must contain at least 2 observations.
     * </li></ul></p>
     *
     * @param mu constant value to compare sample mean against
     * @param sampleStats StatisticalSummary describing sample data
     * @return p-value
     * @throws NullArgumentException if <code>sampleStats</code> is <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double tTest(final double mu, final StatisticalSummary sampleStats)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sampleStats);
        return tTest(sampleStats.getMean(), mu, sampleStats.getVariance(),
                     sampleStats.getN());

    }

    /**
     * Performs a <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm">
     * two-sided t-test</a> evaluating the null hypothesis that the mean of the
     * population from which the dataset described by <code>stats</code> is
     * drawn equals <code>mu</code>.
     * <p>
     * Returns <code>true</code> iff the null hypothesis can be rejected with
     * confidence <code>1 - alpha</code>.  To  perform a 1-sided test, use
     * <code>alpha * 2.</code></p>
     * <p>
     * <strong>Examples:</strong><br><ol>
     * <li>To test the (2-sided) hypothesis <code>sample mean = mu </code> at
     * the 95% level, use <br><code>tTest(mu, sampleStats, 0.05) </code>
     * </li>
     * <li>To test the (one-sided) hypothesis <code> sample mean < mu </code>
     * at the 99% level, first verify that the measured sample mean is less
     * than <code>mu</code> and then use
     * <br><code>tTest(mu, sampleStats, 0.02) </code>
     * </li></ol></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the one-sample
     * parametric t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/sg_glos.html#one-sample">here</a>
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The sample must include at least 2 observations.
     * </li></ul></p>
     *
     * @param mu constant value to compare sample mean against
     * @param sampleStats StatisticalSummary describing sample data values
     * @param alpha significance level of the test
     * @return p-value
     * @throws NullArgumentException if <code>sampleStats</code> is <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public boolean tTest(final double mu, final StatisticalSummary sampleStats,
                         final double alpha)
    throws NullArgumentException, NumberIsTooSmallException,
    OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return tTest(mu, sampleStats) < alpha;

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a two-sample, two-tailed t-test
     * comparing the means of the input arrays.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the two means are
     * equal in favor of the two-sided alternative that they are different.
     * For a one-sided test, divide the returned value by 2.</p>
     * <p>
     * The test does not assume that the underlying popuation variances are
     * equal  and it uses approximated degrees of freedom computed from the
     * sample data to compute the p-value.  The t-statistic used is as defined in
     * {@link #t(double[], double[])} and the Welch-Satterthwaite approximation
     * to the degrees of freedom is used,
     * as described
     * <a href="http://www.itl.nist.gov/div898/handbook/prc/section3/prc31.htm">
     * here.</a>  To perform the test under the assumption of equal subpopulation
     * variances, use {@link #homoscedasticTTest(double[], double[])}.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the p-value depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return p-value for t-test
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double tTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sample1);
        checkSampleData(sample2);
       // System.out.println("Length sample 1: "+sample1.length);
       // System.out.println("Length sample 2: "+sample2.length);
        // No try-catch or advertised exception because args have just been checked
        return tTest(StatUtils.mean(sample1), StatUtils.mean(sample2),
                     StatUtils.variance(sample1), StatUtils.variance(sample2),
                     sample1.length, sample2.length);

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a two-sample, two-tailed t-test
     * comparing the means of the input arrays, under the assumption that
     * the two samples are drawn from subpopulations with equal variances.
     * To perform the test without the equal variances assumption, use
     * {@link #tTest(double[], double[])}.</p>
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the two means are
     * equal in favor of the two-sided alternative that they are different.
     * For a one-sided test, divide the returned value by 2.</p>
     * <p>
     * A pooled variance estimate is used to compute the t-statistic.  See
     * {@link #homoscedasticT(double[], double[])}. The sum of the sample sizes
     * minus 2 is used as the degrees of freedom.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the p-value depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @return p-value for t-test
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double homoscedasticTTest(final double[] sample1, final double[] sample2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sample1);
        checkSampleData(sample2);
        // No try-catch or advertised exception because args have just been checked
        return homoscedasticTTest(StatUtils.mean(sample1),
                                  StatUtils.mean(sample2),
                                  StatUtils.variance(sample1),
                                  StatUtils.variance(sample2),
                                  sample1.length, sample2.length);

    }

    /**
     * Performs a
     * <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm">
     * two-sided t-test</a> evaluating the null hypothesis that <code>sample1</code>
     * and <code>sample2</code> are drawn from populations with the same mean,
     * with significance level <code>alpha</code>.  This test does not assume
     * that the subpopulation variances are equal.  To perform the test assuming
     * equal variances, use
     * {@link #homoscedasticTTest(double[], double[], double)}.
     * <p>
     * Returns <code>true</code> iff the null hypothesis that the means are
     * equal can be rejected with confidence <code>1 - alpha</code>.  To
     * perform a 1-sided test, use <code>alpha * 2</code></p>
     * <p>
     * See {@link #t(double[], double[])} for the formula used to compute the
     * t-statistic.  Degrees of freedom are approximated using the
     * <a href="http://www.itl.nist.gov/div898/handbook/prc/section3/prc31.htm">
     * Welch-Satterthwaite approximation.</a></p>
     * <p>
     * <strong>Examples:</strong><br><ol>
     * <li>To test the (2-sided) hypothesis <code>mean 1 = mean 2 </code> at
     * the 95% level,  use
     * <br><code>tTest(sample1, sample2, 0.05). </code>
     * </li>
     * <li>To test the (one-sided) hypothesis <code> mean 1 < mean 2 </code>,
     * at the 99% level, first verify that the measured  mean of <code>sample 1</code>
     * is less than the mean of <code>sample 2</code> and then use
     * <br><code>tTest(sample1, sample2, 0.02) </code>
     * </li></ol></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li>
     * <li> <code> 0 < alpha < 0.5 </code>
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with
     * confidence 1 - alpha
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public boolean tTest(final double[] sample1, final double[] sample2,
                         final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return tTest(sample1, sample2) < alpha;

    }

    /**
     * Performs a
     * <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm">
     * two-sided t-test</a> evaluating the null hypothesis that <code>sample1</code>
     * and <code>sample2</code> are drawn from populations with the same mean,
     * with significance level <code>alpha</code>,  assuming that the
     * subpopulation variances are equal.  Use
     * {@link #tTest(double[], double[], double)} to perform the test without
     * the assumption of equal variances.
     * <p>
     * Returns <code>true</code> iff the null hypothesis that the means are
     * equal can be rejected with confidence <code>1 - alpha</code>.  To
     * perform a 1-sided test, use <code>alpha * 2.</code>  To perform the test
     * without the assumption of equal subpopulation variances, use
     * {@link #tTest(double[], double[], double)}.</p>
     * <p>
     * A pooled variance estimate is used to compute the t-statistic. See
     * {@link #t(double[], double[])} for the formula. The sum of the sample
     * sizes minus 2 is used as the degrees of freedom.</p>
     * <p>
     * <strong>Examples:</strong><br><ol>
     * <li>To test the (2-sided) hypothesis <code>mean 1 = mean 2 </code> at
     * the 95% level, use <br><code>tTest(sample1, sample2, 0.05). </code>
     * </li>
     * <li>To test the (one-sided) hypothesis <code> mean 1 < mean 2, </code>
     * at the 99% level, first verify that the measured mean of
     * <code>sample 1</code> is less than the mean of <code>sample 2</code>
     * and then use
     * <br><code>tTest(sample1, sample2, 0.02) </code>
     * </li></ol></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The observed array lengths must both be at least 2.
     * </li>
     * <li> <code> 0 < alpha < 0.5 </code>
     * </li></ul></p>
     *
     * @param sample1 array of sample data values
     * @param sample2 array of sample data values
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with
     * confidence 1 - alpha
     * @throws NullArgumentException if the arrays are <code>null</code>
     * @throws NumberIsTooSmallException if the length of the arrays is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public boolean homoscedasticTTest(final double[] sample1, final double[] sample2,
                                      final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return homoscedasticTTest(sample1, sample2) < alpha;

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a two-sample, two-tailed t-test
     * comparing the means of the datasets described by two StatisticalSummary
     * instances.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the two means are
     * equal in favor of the two-sided alternative that they are different.
     * For a one-sided test, divide the returned value by 2.</p>
     * <p>
     * The test does not assume that the underlying population variances are
     * equal  and it uses approximated degrees of freedom computed from the
     * sample data to compute the p-value.   To perform the test assuming
     * equal variances, use
     * {@link #homoscedasticTTest(StatisticalSummary, StatisticalSummary)}.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the p-value depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The datasets described by the two Univariates must each contain
     * at least 2 observations.
     * </li></ul></p>
     *
     * @param sampleStats1  StatisticalSummary describing data from the first sample
     * @param sampleStats2  StatisticalSummary describing data from the second sample
     * @return p-value for t-test
     * @throws NullArgumentException if the sample statistics are <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double tTest(final StatisticalSummary sampleStats1,
                        final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sampleStats1);
        checkSampleData(sampleStats2);
        return tTest(sampleStats1.getMean(), sampleStats2.getMean(),
                     sampleStats1.getVariance(), sampleStats2.getVariance(),
                     sampleStats1.getN(), sampleStats2.getN());

    }

    /**
     * Returns the <i>observed significance level</i>, or
     * <i>p-value</i>, associated with a two-sample, two-tailed t-test
     * comparing the means of the datasets described by two StatisticalSummary
     * instances, under the hypothesis of equal subpopulation variances. To
     * perform a test without the equal variances assumption, use
     * {@link #tTest(StatisticalSummary, StatisticalSummary)}.
     * <p>
     * The number returned is the smallest significance level
     * at which one can reject the null hypothesis that the two means are
     * equal in favor of the two-sided alternative that they are different.
     * For a one-sided test, divide the returned value by 2.</p>
     * <p>
     * See {@link #homoscedasticT(double[], double[])} for the formula used to
     * compute the t-statistic. The sum of the  sample sizes minus 2 is used as
     * the degrees of freedom.</p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the p-value depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">here</a>
     * </p><p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The datasets described by the two Univariates must each contain
     * at least 2 observations.
     * </li></ul></p>
     *
     * @param sampleStats1  StatisticalSummary describing data from the first sample
     * @param sampleStats2  StatisticalSummary describing data from the second sample
     * @return p-value for t-test
     * @throws NullArgumentException if the sample statistics are <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public double homoscedasticTTest(final StatisticalSummary sampleStats1,
                                     final StatisticalSummary sampleStats2)
        throws NullArgumentException, NumberIsTooSmallException,
        MaxCountExceededException {

        checkSampleData(sampleStats1);
        checkSampleData(sampleStats2);
        return homoscedasticTTest(sampleStats1.getMean(),
                                  sampleStats2.getMean(),
                                  sampleStats1.getVariance(),
                                  sampleStats2.getVariance(),
                                  sampleStats1.getN(), sampleStats2.getN());

    }

    /**
     * Performs a
     * <a href="http://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm">
     * two-sided t-test</a> evaluating the null hypothesis that
     * <code>sampleStats1</code> and <code>sampleStats2</code> describe
     * datasets drawn from populations with the same mean, with significance
     * level <code>alpha</code>.   This test does not assume that the
     * subpopulation variances are equal.  To perform the test under the equal
     * variances assumption, use
     * {@link #homoscedasticTTest(StatisticalSummary, StatisticalSummary)}.
     * <p>
     * Returns <code>true</code> iff the null hypothesis that the means are
     * equal can be rejected with confidence <code>1 - alpha</code>.  To
     * perform a 1-sided test, use <code>alpha * 2</code></p>
     * <p>
     * See {@link #t(double[], double[])} for the formula used to compute the
     * t-statistic.  Degrees of freedom are approximated using the
     * <a href="http://www.itl.nist.gov/div898/handbook/prc/section3/prc31.htm">
     * Welch-Satterthwaite approximation.</a></p>
     * <p>
     * <strong>Examples:</strong><br><ol>
     * <li>To test the (2-sided) hypothesis <code>mean 1 = mean 2 </code> at
     * the 95%, use
     * <br><code>tTest(sampleStats1, sampleStats2, 0.05) </code>
     * </li>
     * <li>To test the (one-sided) hypothesis <code> mean 1 < mean 2 </code>
     * at the 99% level,  first verify that the measured mean of
     * <code>sample 1</code> is less than  the mean of <code>sample 2</code>
     * and then use
     * <br><code>tTest(sampleStats1, sampleStats2, 0.02) </code>
     * </li></ol></p>
     * <p>
     * <strong>Usage Note:</strong><br>
     * The validity of the test depends on the assumptions of the parametric
     * t-test procedure, as discussed
     * <a href="http://www.basic.nwu.edu/statguidefiles/ttest_unpaired_ass_viol.html">
     * here</a></p>
     * <p>
     * <strong>Preconditions</strong>: <ul>
     * <li>The datasets described by the two Univariates must each contain
     * at least 2 observations.
     * </li>
     * <li> <code> 0 < alpha < 0.5 </code>
     * </li></ul></p>
     *
     * @param sampleStats1 StatisticalSummary describing sample data values
     * @param sampleStats2 StatisticalSummary describing sample data values
     * @param alpha significance level of the test
     * @return true if the null hypothesis can be rejected with
     * confidence 1 - alpha
     * @throws NullArgumentException if the sample statistics are <code>null</code>
     * @throws NumberIsTooSmallException if the number of samples is &lt; 2
     * @throws OutOfRangeException if <code>alpha</code> is not in the range (0, 0.5]
     * @throws MaxCountExceededException if an error occurs computing the p-value
     */
    public boolean tTest(final StatisticalSummary sampleStats1,
                         final StatisticalSummary sampleStats2,
                         final double alpha)
        throws NullArgumentException, NumberIsTooSmallException,
        OutOfRangeException, MaxCountExceededException {

        checkSignificanceLevel(alpha);
        return tTest(sampleStats1, sampleStats2) < alpha;

    }

    //----------------------------------------------- Protected methods

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

    /**
     * Computes t test statistic for 1-sample t-test.
     *
     * @param m sample mean
     * @param mu constant to test against
     * @param v sample variance
     * @param n sample n
     * @return t test statistic
     */
    protected double t(final double m, final double mu,
                       final double v, final double n) {
        return (m - mu) / FastMath.sqrt(v / n);
    }

    /**
     * Computes t test statistic for 2-sample t-test.
     * <p>
     * Does not assume that subpopulation variances are equal.</p>
     *
     * @param m1 first sample mean
     * @param m2 second sample mean
     * @param v1 first sample variance
     * @param v2 second sample variance
     * @param n1 first sample n
     * @param n2 second sample n
     * @return t test statistic
     */
    protected double t(final double m1, final double m2,
                       final double v1, final double v2,
                       final double n1, final double n2)  {
        return (m1 - m2) / FastMath.sqrt((v1 / n1) + (v2 / n2));
    }

    /**
     * Computes t test statistic for 2-sample t-test under the hypothesis
     * of equal subpopulation variances.
     *
     * @param m1 first sample mean
     * @param m2 second sample mean
     * @param v1 first sample variance
     * @param v2 second sample variance
     * @param n1 first sample n
     * @param n2 second sample n
     * @return t test statistic
     */
    protected double homoscedasticT(final double m1, final double m2,
                                    final double v1, final double v2,
                                    final double n1, final double n2)  {
        final double pooledVariance = ((n1  - 1) * v1 + (n2 -1) * v2 ) / (n1 + n2 - 2);
        return (m1 - m2) / FastMath.sqrt(pooledVariance * (1d / n1 + 1d / n2));
    }

    /**
     * Computes p-value for 2-sided, 1-sample t-test.
     *
     * @param m sample mean
     * @param mu constant to test against
     * @param v sample variance
     * @param n sample n
     * @return p-value
     * @throws MaxCountExceededException if an error occurs computing the p-value
     * @throws MathIllegalArgumentException if n is not greater than 1
     */
    protected double tTest(final double m, final double mu,
                           final double v, final double n)
        throws MaxCountExceededException, MathIllegalArgumentException {

        final double t = FastMath.abs(t(m, mu, v, n));
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final TDistribution distribution = new TDistribution(null, n - 1);
        return 2.0 * distribution.cumulativeProbability(-t);

    }

    /**
     * Computes p-value for 2-sided, 2-sample t-test.
     * <p>
     * Does not assume subpopulation variances are equal. Degrees of freedom
     * are estimated from the data.</p>
     *
     * @param m1 first sample mean
     * @param m2 second sample mean
     * @param v1 first sample variance
     * @param v2 second sample variance
     * @param n1 first sample n
     * @param n2 second sample n
     * @return p-value
     * @throws MaxCountExceededException if an error occurs computing the p-value
     * @throws NotStrictlyPositiveException if the estimated degrees of freedom is not
     * strictly positive
     */
    protected double tTest(final double m1, final double m2,
                           final double v1, final double v2,
                           final double n1, final double n2)
        throws MaxCountExceededException, NotStrictlyPositiveException {

        final double t = FastMath.abs(t(m1, m2, v1, v2, n1, n2));
        final double degreesOfFreedom = df(v1, v2, n1, n2);
        
       // System.out.println("Test: degreesOfFreedom: "+degreesOfFreedom);
        
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final TDistribution distribution = new TDistribution(null, degreesOfFreedom);
        return 2.0 * distribution.cumulativeProbability(-t);

    }

    /**
     * Computes p-value for 2-sided, 2-sample t-test, under the assumption
     * of equal subpopulation variances.
     * <p>
     * The sum of the sample sizes minus 2 is used as degrees of freedom.</p>
     *
     * @param m1 first sample mean
     * @param m2 second sample mean
     * @param v1 first sample variance
     * @param v2 second sample variance
     * @param n1 first sample n
     * @param n2 second sample n
     * @return p-value
     * @throws MaxCountExceededException if an error occurs computing the p-value
     * @throws NotStrictlyPositiveException if the estimated degrees of freedom is not
     * strictly positive
     */
    protected double homoscedasticTTest(double m1, double m2,
                                        double v1, double v2,
                                        double n1, double n2)
        throws MaxCountExceededException, NotStrictlyPositiveException {

        final double t = FastMath.abs(homoscedasticT(m1, m2, v1, v2, n1, n2));
        final double degreesOfFreedom = n1 + n2 - 2;
        // pass a null rng to avoid unneeded overhead as we will not sample from this distribution
        final TDistribution distribution = new TDistribution(null, degreesOfFreedom);
        return 2.0 * distribution.cumulativeProbability(-t);

    }

    /**
     * Check significance level.
     *
     * @param alpha significance level
     * @throws OutOfRangeException if the significance level is out of bounds.
     */
    private void checkSignificanceLevel(final double alpha)
        throws OutOfRangeException {

        if (alpha <= 0 || alpha > 0.5) {
            throw new OutOfRangeException(LocalizedFormats.SIGNIFICANCE_LEVEL,
                                          alpha, 0.0, 0.5);
        }

    }

    /**
     * Check sample data.
     *
     * @param data Sample data.
     * @throws NullArgumentException if {@code data} is {@code null}.
     * @throws NumberIsTooSmallException if there is not enough sample data.
     */
    private void checkSampleData(final double[] data)
        throws NullArgumentException, NumberIsTooSmallException {

        if (data == null) {
            throw new NullArgumentException();
        }
        if (data.length < 2) {
            throw new NumberIsTooSmallException(
                    LocalizedFormats.INSUFFICIENT_DATA_FOR_T_STATISTIC,
                    data.length, 2, true);
        }

    }

    /**
     * Check sample data.
     *
     * @param stat Statistical summary.
     * @throws NullArgumentException if {@code data} is {@code null}.
     * @throws NumberIsTooSmallException if there is not enough sample data.
     */
    private void checkSampleData(final StatisticalSummary stat)
        throws NullArgumentException, NumberIsTooSmallException {

        if (stat == null) {
            throw new NullArgumentException();
        }
        if (stat.getN() < 2) {
            throw new NumberIsTooSmallException(
                    LocalizedFormats.INSUFFICIENT_DATA_FOR_T_STATISTIC,
                    stat.getN(), 2, true);
        }

    }

}
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
/* 
 * KozaFitness.java
 * 
 * Created: Fri Oct 15 14:26:44 1999 
 * By: Sean Luke
 */


/**
 * KozaFitness is a Fitness which stores an individual's fitness as described in
 * Koza I.  Well, almost.  In KozaFitness, standardized fitness and raw fitness
 * are considered the same (there are different methods for them, but they return
 * the same thing).  Standardized fitness ranges from 0.0 inclusive (the best)
 * to infinity exclusive (the worst).  Adjusted fitness converts this, using
 * the formula adj_f = 1/(1+f), into a scale from 0.0 exclusive (worst) to 1.0
 * inclusive (best).  While it's the standardized fitness that is stored, it
 * is the adjusted fitness that is printed out.
 * This is all just convenience stuff anyway; selection methods
 * generally don't use these fitness values but instead use the betterThan
 * and equalTo methods.
 *
 <p><b>Default Base</b><br>
 gp.koza.fitness
 *
 *
 * @author Sean Luke
 * @version 1.0 
 */

@SuppressWarnings("serial")
public class StochasticFitness extends KozaFitness
    {
    
	private double mean;
	private double variance;
	private double [] errArr;
	
               
	public double [] getErrArr(){
		return errArr;
	}
	public void setErrArr(double err[], int length){
		errArr=new double [length];
		for(int i=0;i<length;i++){
			errArr[i]=err[i];
		}
	
	}
	
	public double getMean() {
		return mean;
	}
	public void setMean(double mean2) {
		this.mean = mean2;
	}
	public double getVariance() {
		return variance;
	}
	public void setVariance(double variance) {
		this.variance = variance;
	}
    
     int nTrial=1;   
     
      
  /* public boolean betterThan(StochasticFitness f2, EvolutionState state)
        {
        //return first.fitness.betterThan(second.fitness);
    	    	
    	double subMean = getMean() - f2.getMean();
    	double subStd = Math.sqrt((getVariance() + f2.getVariance())/SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
    	
    	int count = 0;
    	for(int i = 0; i < ec.pta.sfgp.SFProblem.NUMTRAIL; i++)
    	{
            /* Gassian distridution
            double gauss = state.random[0].nextGaussian();
            gauss = gauss * subStd + subMean;
    	
    		if (gauss < 0)
    			count += 1; 
            
            // Student't distribution
            double t = SimpleRNG.GetStudentT(new TTest().df(this.getVariance(), f2.getVariance(), SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER, SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER));
            t = t * subStd + subMean;
            if (t< 0)  
                count += 1; 
                 
    	} 
    	
    	if (count > ec.pta.sfgp.SFProblem.NUMTRAIL/2)
    		return true;
    	else 
    		return false;
    } */
    // Statistics test
   public int betterThan_StatTest(StochasticFitness f, EvolutionState stat)
        { 
           
        double pValue=0;
        
    	// Mann Whitney U Test
        //MannWhitneyUTest t=new MannWhitneyUTest();
        //Returns the asymptotic observed significance level, or p-value
        //pValue=t.mannWhitneyUTest(this.errArr, f.errArr);
        
        // Test with T test
        TTest t1 =new TTest();
        pValue=t1.tTest(this.getErrArr(), f.getErrArr());
        // confidence level 95% => pValue <0.05
       if(pValue<0.05){
      
                if(this.mean<f.mean)
                    return 1; // choose this
                else
                    return 2; // choose f
        }
       else{
           if(this.mean<f.mean)
               return 1;
           else
               return 2; // can't reject H0, can chose this or f
       }
    	
       /*
        double [] t=ztest(this.getMean(),f.getMean(),0,this.getVariance(),f.getVariance(),this.getErrArr().length,f.getErrArr().length);
        if(t[1]<ec.select.ProbabilisticTourSelection.alpha){
            if(this.getMean()<f.getMean())
               return 1; // chose f1
            else
                return 2; // chose f2
        }
        else
            {
           if(this.mean<f.mean)
               return 1;
           else
               return 2; // can't reject H0, can chose this or f
       }
        //return oneTailedZTest(f1,f2, 0.95);
        
	*/
    } 
 // Two sample one tailed Z test tu viet, level: muc do tin cay
   // because size sample>30 then using Ztest - sach thay gui - nghien cuu thuc nghiem
   public int oneTailedZTest(StochasticFitness f, double level){
       double var=0, sD, t, lowCriticalValue=0, highCriticalValue=0;
       
       var=(this.variance+f.variance)/(SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER-1);
       sD=Math.sqrt(var/SFProblem.NUMBOOTSAMPLE_AFTER_REMOVE_OUTLIER);
       t=(this.mean-f.mean)/sD;
       // Do tim t trong bang degrees of freedom, trong truong hop nay size cua mau>30, dung Ztest
       if(level==0.95){ // confidence level = 95% ~ alpha = (1-0.95)=0.05
           lowCriticalValue=-1.64;  // one tailed
           highCriticalValue=1.64;  // one tailed
       }
       if (level==0.99){// confidence level = 99% ~ alpha=(1-0.99)=0.01
           lowCriticalValue=-2.33;
           highCriticalValue=2.33;
       }
       if (level==0.9){ // confidence=90% ~ alpha = 1 - 0.9 = 0.1
           lowCriticalValue=-1.28;
           highCriticalValue=1.28;
       }
       if(t<lowCriticalValue)   return 1; // Reject Ho and f.mean<g.mean, chose f
       if(t>highCriticalValue)  return 2; // Reject Ho and f.mean>g.mean, chose g
       return 3; // no Reject H0
   }
     // NEW: from (python): TODO return z score and pvalue. X1,X2 : 2 samples, mudiff =0, n1, n2: size of sample
    public static double[] ztest(double meanX1, double meanX2, int mudiff, double var1, double var2, int n1, int n2){
        double tam[]=new double[2];
        NormalDistribution d=new NormalDistribution();
        double pooledSE = Math.sqrt(var1/n1 + var2/n2);
        double z = ((meanX1 - meanX2) - mudiff)/pooledSE; // mufiff=0=mean cua phan bo N(0,1)
        // pval = 2*(1 - norm.cdf(abs(z)))
        double pval = 2*(1 - d.cumulativeProbability(Math.abs(z)));
        tam[0]=z;
        tam[1]=pval;
        return tam;
        //z, p = twoSampZ(28, 33, 0, 14.1, 9.5, 75, 50)
        //print (z, p)
    }    
    // so sanh chon thang tot nhat nhu Thay da goi y
    public boolean betterThan_mean(final StochasticFitness _fitness)
        {
        return _fitness.fitness() < fitness();
        }
    }
