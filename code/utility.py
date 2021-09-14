
import _PyPacwar
import numpy
import numpy as np


def get_score(g1,g2):

    death_punishment = -2 #  punishment score if it totally loses (zero pacman left)
    loss_punishment = -1 #  punishment score if it loses
    total_win_award = 4 # add this award while totally win

    death_punishment, loss_punishment, total_win_award = 0, 0, 0

    death_punishment =0
    loss_punishment =0
    total_win_award =0.0

    (r, c1,c2) = _PyPacwar.battle(g1,g2)
    if( c1 ==0 or c2 ==0):
        if( r < 100 ): sh, sl = 20 + total_win_award ,0 + death_punishment
        elif(r <200): sh,sl = 19 + total_win_award ,1+ death_punishment
        elif(r < 300): sh,sl = 18 + total_win_award ,2+ death_punishment
        else: sh,sl = 17 + total_win_award ,3 + death_punishment
    else:
        ratio = min(c1,c2)/max(c1,c2)
        if( ratio <= 0.1 ): sh, sl = 13,7 + loss_punishment
        elif( ratio <= 1.0/3.0): sh,sl = 12,8 + loss_punishment
        elif( ratio <= 1.0/1.5): sh,sl = 11,9 + loss_punishment
        else: sh, sl = 10 + loss_punishment , 10 + loss_punishment


    if (c1 < c2 ):
        return sl, sh
    else:
        return sh, sl







def get_score_vs_opponents(gene, opponenets):
    score = 0.0
    for g2 in opponenets:
        s, _ = get_score(gene,g2)
        score += s
    return score/len(opponenets)

def str_to_gene_list(str):
    return [int(x) for x in str]

def read_gene_list(path):
    file = open(path)
    gene = []
    while(1):
        line = file.readline()

        if not line:
            break
        g = line.rstrip('\n')
        g = str_to_gene_list(g)
        gene.append(g)
    return gene

def get_random_gene():
    # randomly generate a gene
    rep = [0]*50
    for i in range(0,50):
        rep[i] = int( numpy.random.randint(0,4))

    return rep


def evaluate_diversity(population):
    # evaluate the diversity of population
    population = np.asarray( population )
    std = np.std(population, axis = 0 )
    return np.mean(std), np.max(std), std



if __name__ == "__main__":
    opp = [ [1] * 50, [2] * 50, [3] * 50]
    g = "00100003110102203133111311111111111111210331121331"
    opp.append( str_to_gene_list(g) )
    g = "10113212331112111301111111111112311111131321211021"
    opp.append(str_to_gene_list(g))

    test_case = read_gene_list( "fnn_gene.txt" )
    test_case = [str_to_gene_list("00020000111122203333321321321121321321103130301301")]
    print(len("00020000111122203333321321321121321321103130301301"))

    for i in range(0,len(test_case)):
        print("test case :", i)
        g1 = test_case[i]
        print(g1)
        print("score: ")
        for g2 in opp:
            c1,c2 = get_score(g1,g2)
            print(c1, end = ' ')
        print('\n')

