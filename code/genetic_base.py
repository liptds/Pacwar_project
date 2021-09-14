
import _PyPacwar
import numpy as np
from utility import *



class genetic:
    """
    genetic algorithm for pac war projects
    """

    def __init__(self, config, fitness, x0 = None ):
        self.config = config
        if x0 is None: # randomly init the initial population
            self.population_0 =[ get_random_gene() for i in range(0,self.config['population_size'])]
        else:
            self.population_0 = x0
        self.fitness = fitness

    def update_fitness(self, fitness): # update fitness function
        self.fitness = fitness # acting on the whole population



    def selection(self, population, score):
        # do the selection according to the score
        # score = [ score_i, ..]
        # the population is sorted according to the score
        mating_pool = [] # mating pool
        size_of_mating_pool = int( self.config['mating_pool_ratio'] * len(population) )
        for i in range(0, size_of_mating_pool ):
            mating_pool.append(population[i])
        return mating_pool, score[0:size_of_mating_pool]

    def crossover(self, mating_pool, type = 'bit'):
        # type = bit : crossover for each bit
        children = []
        population_size = self.config['population_size']
        segment_len = [4, 16, 3, 3, 12,12 ] # U,V,W,X,Y,Z
        for i in range(population_size):
            par_1 = mating_pool[np.random.randint(0, len(mating_pool))]
            par_2 = mating_pool[np.random.randint(0, len(mating_pool))]
            if(type == 'bit'): # crossover for each bit
                child = [0] * len(par_1)
                for digit in range(0, len(par_1)):
                    if (np.random.rand() < 0.5):
                        child[digit] = par_1[digit]
                    else:
                        child[digit] = par_2[digit]
            elif( type == 'segment'): # pit a whole segement, such as U segment, X segment ,...
                child = [0] * len(par_1)
                pt = 0
                for len_ in (segment_len):
                    if (np.random.rand() < 0.5):
                        child[pt:pt+len_] = par_1[pt:pt+len_]
                    else:
                        child[pt:pt + len_] = par_2[pt:pt + len_]
                    pt += len_
            children.append(child)


        return children


    def random_mutation_each_gene(self, gene, type = 'bit' ):
        """
        type = bit
        random pick one element in rep_old and randomly set its value
        :param rep_old:
        :return:
        """
        for pos in range(0, len(gene)):
            if( type == 'bit'):
                if (np.random.rand() < self.config['mutation_prob']):
                    value = int(np.random.randint(0, 4)) # 0.25 probability to keep original gene
                    gene[pos] = value
                    # if (value >= gene[pos]):
                    #     gene[pos] = (value + 1)
                    # else:
                    #     gene[pos] = value

        return gene


    def evolution(self, population , score = None, mutation_type = 'bit', crossover_type = 'bit'):
        if( score is None):
            score = self.fitness(population)
            score = [ (s, i) for i,s in enumerate(score)]
        mating_pool,_ = self.selection(population, score)
        children = self.crossover(mating_pool,crossover_type)
        new_population = [self.random_mutation_each_gene(child, mutation_type) for child in children]
        return new_population

    def evaluate(self, population):
        score = self.fitness(population)
        score = [(s, i) for i, s in enumerate(score)]
        score = sorted(score, key=lambda x: -x[0])
        population = [population[x[1]] for x in score]
        score = [x[0] for x in score]
        return population, score