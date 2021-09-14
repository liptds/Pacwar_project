import numpy as np
from utility import *


class fitness_fun:
    """
    construct fitness function
    """

    def __init__(self, config, pools = []):
        self.pools = pools
        self.pool_set = set()
        self.config = config

    def read_gene(self, filename):
        """
         read gene to poolss from filename
        :param filename:
        :return:
        """

        file = open(filename)
        while(1):
            line = file.readline()
            if not line:
                break
            line = line.rstrip('\n')
            gene = str_to_gene_list(line)
            self.add_gene_to_pool(gene)

        file.close()
    def write_gene_to_pools(self, filename):
        file = open(filename,'w+')
        for gene in self.pools:
            gene = "".join( [str(i) for i in gene])
            gene = gene + '\n'
            file.writelines(gene)
        file.close()

    def get_score_matrix(self, population_1, population_2 = None):
        """
        score[i,j] = score of population_1[i] after fighting with population_2[j]
        :param population_1:
        :param population_2:
        :return:
        """

        if( population_2 is None):
            score = np.zeros((len(population_1), len(population_1)))
            for i in range(0, len(population_1)):
                for j in range(i + 1, len(population_1)):
                    si, sj = get_score(population_1[i], population_1[j])
                    score[i, j] = si
                    score[j, i] = sj
        else:
            score = np.zeros((len(population_1), len(population_2)))
            for i in range(0, len(population_1)):
                for j in range(0, len(population_2)):
                    si, sj = get_score(population_1[i], population_2[j])
                    score[i, j] = si
        return score

    def tournament_fitness(self, population):
        # generate fitness function by tournament
        score = self.get_score_matrix(population)
        score = np.mean(score, axis=1)
        return score

    def opponent_fitness(self, population, if_random = True):
        """
        fighting against pools
        :param if_random:  = True randomly pick n opponents from pools
        :return:
        """
        if( len(self.pools) == 0):
            return np.zeros(population.shape)
        if( if_random):
            index_ = np.random.choice(len(self.pools), min(len(self.pools), self.config['n_enemy']), replace=False)
            enemy = [self.pools[i] for i in index_]
        else:
            enemy = self.pools
        score = self.get_score_matrix(population, enemy)
        score = np.mean(score, axis=1)
        return score

    def hybrid_fitness(self, population, weight = (0.5,0.5), if_random =True):
        """
        w[0] * tournament_fitness + w[1] * opponent_fitness
        :param population:
        :param weight:
        :param if_random:
        :return:
        """
        if( weight[0] > 1e-3):
            s1 = self.tournament_fitness(population)
        else:
            s1 = 0.0
        if(weight[1] > 1e-3):
            s2 = self.opponent_fitness(population, if_random)
        else:
            s2 = 0.0
        return s1 * weight[0] + s2 * weight[1]

    def prune_pools(self, size):
        # prune enemey pools to size
        if( len(self.pools) < size ):
            return
        score = self.tournament_fitness(self.pools)
        score = [ (s,i) for i,s in enumerate(score)]
        score = sorted(score, key = lambda x:-x[0])
        pools_new = []
        set_new = set()
        for i in range(0,size):
            gene = self.pools[score[i][1]]
            code = "".join( [str(i) for i in gene ])
            set_new.add(code)
            pools_new.append(gene)
        self.pools = pools_new
        self.pool_set = set_new

    def add_gene_to_pool(self, gene):
        code = "".join([str(i) for i in gene])
        if( code in self.pool_set):
            return
        else:
            self.pool_set.add(code)
            self.pools.append(gene)

