'''
Created on Dec 31, 2001

@author: hanu
'''
import glob, os, sys, math, csv, codecs
from operator import itemgetter


problems = [
                #"nguyen-4",
#                 "keijzer-1",
#                 "keijzer-4",
#                 "keijzer-6",
#                 "keijzer-7",
# #                 "keijzer-8",
#                 "keijzer-9",
#                 "keijzer-10",
#                 "keijzer-11",
#                 "keijzer-12",
#                 "keijzer-13",
#                 "keijzer-14",
#                 "keijzer-15",
# #                "nguyen-8",
#                 "r1",
#                 "r2",
#                 "r3",
#                
#               "casp",
#               "slump_test_FLOW",
#               "slump_test_Compressive",
#               "slump_test_SLUMP",
#                 
#               "airfoil_self_noise",
#               "ccpp", 
#               "concrete", 
#               "winequality-red",
#               "winequality-white", 
#               "wpbc"

#                "F2", 
#              	"F4", 
#		"F5", 
 #		"F7", 
 #		"F8",
		"F9",
#		"F10",
#		"F11",
#		"bupa",
#		"Census6",
#		"No2",

                ]

operators = [
               "sc", 
                 #'sgxe', 
                 #'sgxm', 
                 #"sgxe+sgmr", "sgxm+sgmr",  #10% mutation
                 #'sgxesc20', 'sgxesc30', 'sgxmsc20', 
                 #'sgxmsc30',
                 #'sgxesc', 
                 #'sgxmsc',
                 #"sgxesc+sgmrsc", "sgxmsc+sgmrsc",
                 #'rdo',
                 #"agx"
		#'bgp',
		'bvgp',
		#'bvgp_star',
                'sfgp'
                 ]
    
    
numgens = 300
    

def getResult():
    root = '/home/pta/projects/ECJ-Regression/out/sc50'
    
    result = os.path.join(root, 'result.csv')
    fs = csv.writer(open(result,'w'), quoting=csv.QUOTE_ALL)
    rows = ['problem', 'fitness', 'fittest'] 
    for problem in problems:
        rows.append([problem])
        best_fitness = 1000
        
        dir = os.path.join(root, problem)
        files = glob.glob(dir+'/*.stat')

        fittests = []
        
        for file in files:
            lines = codecs.open(file).readlines()
            vals = lines[-1].split()
            t = float(vals[-6]) 
            if( t < best_fitness):
                best_fitness = t
                    
            fittests.append(float(vals[-1]))
            
        rows[-1].append(best_fitness)
        
        fittests.sort()
        rows[-1].append(fittests[len(fittests)/2])
        
    for row in rows:
        fs.writerow(row)

def getDiversity(dir):
    result = dir + '/diversity.csv'
    fs = csv.writer(open(result, 'w'), quoting=csv.QUOTE_ALL)
    files = glob.glob(dir + '/*.stat')
    average = [0]*50 #for 50 generation
    for file in files:
        vals = []
        print file
        gens = codecs.open(file).readlines()
        for i in xrange(len(gens)):
            gen = gens[i]
            temp = gen.strip().split(' ')[-2]
            vals.append(float(temp))
            average[i] += float(temp)
        fs.writerow(vals)
    
    average = [item / len(files) for item in average]
    fs.writerow(average)
    
    
def getTrainingFitness(dir):
    result = dir + '/training.fitness.csv'
    fs = csv.writer(open(result, 'w'), quoting=csv.QUOTE_ALL)
    files = glob.glob(dir + '/*.stat')
    for file in files:
        vals = []
        print file
        gens = codecs.open(file).readlines()
        for gen in gens:
            temp = gen.strip().split(' ')[-6]
            vals.append(temp)
        fs.writerow(vals)

def getTestingFitness(dir):
    result = dir + '/testing.fitness.csv'
    fs = csv.writer(open(result, 'w'), quoting=csv.QUOTE_ALL)
    files = glob.glob(dir + '/*.stat')
    for file in files:
        vals = []
        print file
        gens = codecs.open(file).readlines()
        for gen in gens:
            vs = gen.strip().split()
            temp = vs[-1]
            vals.append(temp)
        fs.writerow(vals)
        

def getAllResult(root, operator, problem):
    #operators = ['sc', 'ssc', 'mssc', "gsgp", "gsgp+sc20", "gsgp+ssc20", "gsgp+mssc20", "gsgp+sc30", "gsgp+ssc30", "gsgp+mssc30"]


    
#    numcolumns = 26
    
    #regression
    # SC, BVGP:
    # 0: generation
    # 6: evaluation time
    # 8: average number of nodes per individual this generation 
    # -3: fittest of the best on gen
    # -9: fitness of the best on gen
    
    indexColumns = [0, 6, 8, -9, -3]
    
    #SFStatistics
    # 0: generation
    # 6: evaluation time
    # 8: average number of nodes per individual this generation
    # -1 fittest of the best on gen
    # -5 variance of the best on gen
    # -6 mean of the best on gen
    # -7 fitness of the best on gen
    if operator == 'sfgp':
        indexColumns = [0, 6, 8, -7, -6, -5, -1]
        
    
    numOfCols = len(indexColumns) + 2
    
    
    average=[]
    for i in xrange(numgens):
        average.append([0]*numOfCols)
        
        
            
    dir = os.path.join(root, operator, problem)
    files = glob.glob(dir + '/*.stat')
    
    numruns = len(files)
    
    # for calculate median on training and testing
    fitness = []
    fittest = []
    for i in xrange(numgens):
        runs1 = []
        runs2 = []
        for j in xrange(numruns):
            runs1.append(0)
            runs2.append(0)
        
        fitness.append(runs1)
        fittest.append(runs2)
    
    # for each run
    for j in xrange(numruns):
        file = files[j]
#        print file
        #read all lines (generations) of this run
        gens = open(file).readlines()
        
        # for each generation
        for i in xrange(numgens):
            gen = gens[i]
            gen = gen.replace('[',"").replace("]","")
            xs = gen.strip().split()
            
            # store fitness and fittest of the best individual for cal median
            # get fitness of the best individual on this generation
            fitness_idx = -9
            fittest_idx = -3
            
            if operator == 'sfgp':
                fitness_idx = -7
                fittest_idx = -1
            
            fitness[i][j] = float(xs[fitness_idx])
            
            # get fittest of the best individual
            fittest[i][j] = float(xs[fittest_idx])
            
            for k in xrange(len(indexColumns)): 
                average[i][k] += float(xs[indexColumns[k]])# / numruns;
                        
            
    for i in xrange(numgens):
        fittest[i].sort();
        fitness[i].sort()
        
        average[i][-2] = fitness[i][numruns/2] * numruns # de sau chia cho numberuns
        average[i][-1] = fittest[i][numruns/2] * numruns # de sau chia cho numberuns
                
                                        
    
    for i in xrange(len(average)):
        for j in xrange(len(average[i])):
            average[i][j] = average[i][j] / numruns

    #write result
    #regression
            
#     f = csv.writer(open(os.path.join('/home/pta/projects/ECJ-Regression/out/', operator, problem + ".csv"), 'w'), quoting=csv.QUOTE_ALL)
# 
#     for i in xrange(numgens):
#         f.writerow(average[i])
        
    return average
                    
def getAllResultClassification(root):
    #operators = ['sc', 'ssc', 'mssc', "gsgp", "gsgp+sc20", "gsgp+ssc20", "gsgp+mssc20", "gsgp+sc30", "gsgp+ssc30", "gsgp+mssc30"]
    operators = [
                 #"sc", 
                 #'sgxe', 'sgxm', 
                 #"sgxe+sgmr", "sgxm+sgmr",  #10% mutation
                 #'sgxesc20', 'sgxesc30', 'sgxmsc20', 
                 #'sgxmsc30',
                 
                 #'sgxesc', 
                 #'sgxmsc',
                 #"sgxesc+sgmrsc", "sgxmsc+sgmrsc",
                #'rdo',
                #"agx"
                
                 ]
    
    problems = [
              

                #CLASSIFICATION
                "data_banknote_authentication",
                 "breast-cancer-wisconsin",
                 "wdbc",
                 "EEGEyeState",
                 "haberman",
                 "magic04",

                ]
    
    numgens = 100
#    numcolumns = 26
    
   
    #classification
    indexColumns = [0, 1, 2, -14, -5, -4, -3, -2, -1]
    
    
    numOfCols = (len(indexColumns) + 1) * len(operators)
    
    
    for problem in problems:
        print problem, 
        average=[]
        for i in xrange(numgens):
            average.append([0]*numOfCols)
        
        for operatorIndex in xrange(len(operators)):
            operator = operators[operatorIndex]
            print operator
            
            dir = os.path.join(root, operator, problem)
            files = glob.glob(dir + '/*.stat')
            
            numruns = len(files)
            
            fittest = []
            for i in xrange(numgens):
                runs = []
                for j in xrange(numruns):
                    runs.append(0)
                
                fittest.append(runs)
            
            for j in xrange(numruns):
                file = files[j]
                #print file
                gens = open(file).readlines()
                for i in xrange(numgens):
                    gen = gens[i]
                    gen = gen.replace('[',"").replace("]","")
                    xs = gen.strip().split()
                    
                    fittest[i][j] = float(xs[-1])
                    
                    for k in xrange(len(indexColumns)):
                        average[i][operatorIndex*(len(indexColumns) + 1) + k] += float(xs[indexColumns[k]]) / numruns;
                        
            #get median in fittest
#            k = operatorIndex*(len(indexColumns) + 1) + len(indexColumns)-1
            
#            for i in xrange(numgens):
#                fittest[i].sort(cmp=None, key=None, reverse=False);
#                average[i][k] = fittest[i][numruns/2]
                    
                                        
            operatorIndex += 1;
            
            
        #write result
        #classification
        f = csv.writer(open(os.path.join('/home/tuananh/Dropbox/PPSN14/classification/AddSubMulDivSinCosSqrtSquare', problem + ".csv"), 'w'), quoting=csv.QUOTE_ALL)
        
        for i in xrange(numgens):
            f.writerow(average[i])          
            
                            
                            
def getDataForStatisticalTesting(root, opers, probs):
    
    NUMRUNS = 100
    NUMGENS = 300
    
    for problem in probs:
	print problem
        
        result = []
        for i in xrange(NUMRUNS):
            result.append([])
            
        for operator in opers:
            print operator
            
            fitness_idx = -9
            fittest_idx = -3
            
            if operator == 'sfgp':
                fitness_idx = -7
                fittest_idx = -1
                
            for i in xrange(NUMRUNS):
                #read file and get all lines (generations)
                fname = os.path.join(root, operator, problem, "job." + str(i) + ".out.stat")
#		print fname
                gens = open(fname, 'rb').readlines()

                l = []
                for gen in gens:
                    gen = gen.replace('[',"").replace("]","")
                    xs = gen.strip().split()
                    
                    l.append([])
                    l[-1].append(int(xs[0])) #gen
                    l[-1].append(float(xs[fitness_idx]))
                    l[-1].append(float(xs[fittest_idx]))
                    
                # sort l by fitness
                l = sorted(l,key=itemgetter(1))
                
                # run, fittest of best individual, fittest of median
                result[i] = result[i] + [i, l[0][2], l[NUMGENS/2][2]]
                
        
        csvOut = os.path.join(root, problem + ".stat.csv")
        out = csv.writer(open(csvOut, 'wb'), quoting=csv.QUOTE_ALL)
        for row in result:
            out.writerow(row)
    
def getThuongResult():
    '''
    Chi Thuong
    '''
    root = '/home/thuongpt/ECJ-Regression/out'
    for problem in problems:
        print problem
        result = []
        for operator in operators:
            print operator
            
            r = getAllResult(root, operator, problem)
            if len(result) == 0:
                result = r
            else:
                for i in xrange(len(r)):
                    result[i] = result[i] + r[i]
        
        # write output
        csvOut = os.path.join(root, problem + ".csv")
        out = csv.writer(open(csvOut, 'wb'), quoting=csv.QUOTE_ALL)
        for row in result:
            out.writerow(row)
    
                                
if __name__ == '__main__':
#     dir = '/home/pta/projects/ECJ-Regression/out/sc/keijzer-1'
    
    
#     getResult()
    
    #getDiversity(dir)
    
#     getTrainingFitness(dir)
    
#     getTestingFitness(dir)
    
    #regression
    
    getThuongResult()   

    getDataForStatisticalTesting('/home/thuongpt/ECJ-Regression/out', operators, problems)
            
    #classification
    #getAllResultClassification('/home/tuananh/Documents/projects/ECJ-Classification/out/uci2')
    
    
#     mergeCSVFiles()
    print 'DONE'
