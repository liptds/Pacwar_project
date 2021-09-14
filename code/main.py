
from genetic_base import *
from utility import *
from fitness_function import *
import matplotlib.pyplot as plt
import numpy as np
import random
from test import main_test
from vae import vae_generator
from vae import VAE
import pandas as pd

if __name__ == '__main__':
    config = {}
    config['mating_pool_ratio'] = 0.1 # put 0.1 * population_size gene into mating pool
    config['population_size'] = 100
    config['mutation_prob'] = 0.1 # mutation prob
    config['n_enemy'] = 100

    x0 = vae_generator("vae.ckpt", int( config['population_size']*0.01 ), scale = 4)
    while( len(x0) < config['population_size']):
        x0.append( get_random_gene() )

    GA = genetic(config, x0)
    fit_fun = fitness_fun(config)
    fit_fun.read_gene('gene_pools.txt')

    fitness = lambda x: fit_fun.hybrid_fitness(x, weight=(0.0,1.0))
    GA.update_fitness(fitness)
    population = GA.population_0
    score = None
    best_score = (-float('inf'), -float('inf'))
    best_sol = None

    high_ = []
    std_ = []

    for i in range(0, 300):
        print(" ----------------------------iteration: ",i, "--------------------------")
        p = random.uniform(0.0,1.0)
        if( p<0.9):
            population= GA.evolution(population, score, crossover_type='segment')
        else:
            population = GA.evolution(population, score, crossover_type='bit')
        population, score = GA.evaluate(population)

        if( i%40 == 0 and i>10 and False ):
            fit_fun.add_gene_to_pool(population[0]) # add new gene to enemy pool
        s = (score[0], np.mean(score))
        if( s>best_score):
            best_sol = population[0]
            best_score = s

        print("Highest score =", s[0], " mean score = ", s[1] ," best score:", best_score)

        high_.append(s[0])


        winner = population[0]
        print("winner:", "".join([str(i) for i in winner]))
        print("best sol:", "".join([str(i) for i in best_sol]))
        if(i%1==0):
            mean_std, max_std, std = evaluate_diversity(population)
            print("mean, max std", mean_std,max_std)
            # plt.plot(std,'-o')
            # plt.show()
            std_.append(mean_std)

            if( max_std < 0.1):
                break

    dict = {"high_score":np.asarray(high_), "std":np.asarray(std_)}
    df = pd.DataFrame(dict)
    df.to_csv("rand.csv")




    fit_fun.write_gene_to_pools("new_gene_pools.txt")
    winner = "".join([str(i) for i in winner])
    print("\n final solution:")
    print(f"winner score {s[0]}, best score {best_score[0]}")
    print("winner:", "".join([str(i) for i in winner]))
    print("best sol:", "".join([str(i) for i in best_sol]))

    print('\n\n')
    main_test("".join([str(i) for i in best_sol]))
    print("best sol:", "".join([str(i) for i in best_sol]))
    #
    # file = open("winner.txt", 'w+')
    # file.writelines(file)
    # file.close()

