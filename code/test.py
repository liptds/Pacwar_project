import sys
import pandas as pd
import _PyPacwar
from utility import *

def get_score_org(g1,g2):


    death_punishment, loss_punishment, total_win_award = 0, 0, 0


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

def test_single(gene):

    enemy_list = read_gene_list("test_pools.txt")
    win_num = 0
    lose_num = 0
    score = 0
    c = 0
    copp = 0
    for i, enemy in enumerate(enemy_list):
        (rounds, c1, c2) = _PyPacwar.battle(gene, enemy)
        s1, s2 = get_score_org(gene, enemy)
        if (c1 >= c2):
            win_num += 1
        else:
            lose_num += 1
        score += s1
        c += c1
        copp += c2

    return [score, win_num/(win_num+lose_num), c, copp ]
def test_all(path):
    file = open(path)
    gene = []
    while (1):
        line = file.readline()
        if not line:
            break
        line = line.rstrip('\n')
        gene.append(str_to_gene_list(line))

    score = []
    for g in gene:
        score.append( test_single((g)))

def main_test(gene_opt):



    gene_opt = str_to_gene_list(gene_opt)

    enemy_list = read_gene_list("test_pools.txt")

    print("------------------------------")
    (rounds, c1, c2) = _PyPacwar.battle(gene_opt, [1] * 50)
    print("all one ")
    print("Number of rounds:", rounds )
    print("Opt PAC-mites remaining:", c1)
    print("Test PAC-mites remaining:", c2)
    print("------------------------------")
    (rounds, c1, c2) = _PyPacwar.battle(gene_opt, [2] * 50)
    print("all two ")
    print("Number of rounds:", rounds)
    print("Opt PAC-mites remaining:", c1)
    print("Test PAC-mites remaining:", c2)
    print("------------------------------")
    (rounds, c1, c2) = _PyPacwar.battle(gene_opt, [3] * 50)
    print("all three ")
    print("Number of rounds:", rounds)
    print("Opt PAC-mites remaining:", c1)
    print("Test PAC-mites remaining:", c2)

    win_num = 0
    lose_num = 0
    score = 0
    c=0
    copp = 0
    for i, enemy in enumerate(enemy_list):
        (rounds, c1, c2) = _PyPacwar.battle(gene_opt, enemy)
        s1, s2 = get_score_org(gene_opt, enemy)
        print("------------------------------")
        print("enemy ",i)
        print("Number of rounds:", rounds)
        print("Opt PAC-mites remaining:", c1)
        print("Test PAC-mites remaining:", c2)
        if( c1 >= c2):
            win_num += 1
        else:
            lose_num += 1
        score += s1
        c+=c1
        copp += c2

    print("\n\n")
    print(f"win rate {win_num/(win_num+lose_num)}, average score {score/(win_num + lose_num)}, average remaining pac {c/(win_num+lose_num)}, opp remaining {copp/(win_num + lose_num)}")


if __name__ == "__main__":
    gene_opt = "03130000300303230333112122111121122122121130120131"
    main_test(gene_opt)

