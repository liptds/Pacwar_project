import _PyPacwar
import numpy as np
import time
import argparse

def get_score(g1,g2):
    # score of g1 vs g2
    # return score 1 score 2
    (r, c1,c2) = _PyPacwar.battle(g1,g2)
    if( c1 ==0 or c2 ==0):
        if( r < 100 ): sh, sl = 20,0
        elif(r <200): sh,sl = 19,1
        elif(r < 300): sh,sl = 18,2
        else: sh,sl = 17,3
    else:
        ratio = min(c1,c2)/max(c1,c2)
        if( ratio <= 0.1 ): sh, sl = 13,7
        elif( ratio <= 1.0/3.0): sh,sl = 12,8
        elif( ratio <= 1.0/1.5): sh,sl = 11,9
        else: sh, sl = 10, 10

    if (c1 < c2 ):
        return sl, sh
    else:
        return sh, sl

def get_score_matrix(rep_list):
    """
    get score matrix
    score[i][j] = score of rep[i] ( rep[i] vs rep[j] )
    :param rep_list:
    :return:
    """
    gene_list = [rep_to_gene(rep) for rep in rep_list]
    score = np.zeros((len(gene_list), len(gene_list)))

    for i in range(0, len(gene_list)):
        for j in range(i + 1, len(gene_list)):
            si, sj = get_score(gene_list[i], gene_list[j])
            score[i, j] = si
            score[j, i] = sj
    return score

def fitness( rep, enemy_list ):
    # rep vs all the enemey
    score = sum([ get_score(gene, rep)[1] for gene in enemy_list ])/len(enemy_list)
    return score

def read_gene(filename):
    """
     read gene to poolss from filename
    :param filename:
    :return:
    """

    file = open(filename)
    gene_list = []
    while(1):
        line = file.readline()
        if not line:
            break
        line = line.rstrip('\n')
        gene = [int(x) for x in line]
        gene_list.append(gene)
    file.close()
    return gene_list

def neighbour(rep,enemy_list):
    rep_initial = rep[:]
    neighbors = [rep_initial]
    for i in range(0, len(rep)): #Each digit in the gene
        for plus in range(1, 4): # Change for each digit
            temp = rep[:]
            temp[i] = (temp[i] + plus) % 4
            neighbors.append(temp)
    for i in range(0,len(rep)-1):
        for j in range(i+1,len(rep)):
            for plus1 in range(1, 4):
                for plus2 in range(1,4):
                    temp = rep[:]
                    temp[i] = (temp[i] + plus1) % 4
                    temp[j] = (temp[j] + plus2) % 4
                    neighbors.append(temp)
    print(len(neighbors))
    score = [fitness(rep, enemy_list) for rep in neighbors]
    score = np.asarray(score)
    print(score)
    idx = np.argmax(score)

    return idx, score[idx], neighbors[int(idx)]

def crossover(gene1, gene2,enemy_list):
    children = [gene1,gene2]
    population_size = 400
    segment_len = [4, 16, 3, 3, 12, 12]  # U,V,W,X,Y,Z
    for i in range(population_size):
        par_1 = gene1
        par_2 = gene2
        if np.random.rand()<= 1:  # crossover for each bit
            child = [0] * len(par_1)
            for digit in range(0, len(par_1)):
                if (np.random.rand() < 0.5):
                    child[digit] = par_1[digit]
                else:
                    child[digit] = par_2[digit]
        else:  # pit a whole segement, such as U segment, X segment ,...
            child = [0] * len(par_1)
            pt = 0
            for len_ in (segment_len):
                if (np.random.rand() < 0.5):
                    child[pt:pt + len_] = par_1[pt:pt + len_]
                else:
                    child[pt:pt + len_] = par_2[pt:pt + len_]
                pt += len_
        children.append(child)
    print(len(children))
    score = [fitness(rep, enemy_list) for rep in children]
    score = np.asarray(score)
    # print(score)
    idx = np.argmax(score)
    return idx, score[idx], children[int(idx)]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--type', type=str, default='polish', help='polish or crossover')
    # opt = parser.parse_args()

    start = time.time()
    gene_list = read_gene('test_pool.txt')
    # gene_to_polish1 = '03130000000303230333112122111121122122121130121131'
    gene_to_polish1 = '03130000300303230333112122111121122122121130120131' # after polish from the first
    # gene_to_polish2 = '00020000010020220030121123123123123323223313113313'
    rep1 = [int(x) for x in gene_to_polish1]
    # rep2 = [int(x) for x in gene_to_polish2]
    idx, score, best_gene = neighbour(rep1,gene_list) #search 2 neigbours
    # idx, score, best_gene=crossover(rep1,rep2,gene_list) #cross over 2 diversity good genes
    print(score)
    best_gene = "".join([str(i) for i in best_gene])
    print(best_gene)
    end = time.time()
    print(end-start)
