# Genetic Algorithm + final improvement:

1. GSAT
FindGene_GSAT.py Initial try on GSAT algorithms to find the best gene. Only keep 1 gene at a time which improve not as quick and robust as the Genetic algorithm.



2. Genetic Algorithm (GA): 
main.py: Main program, run ga evolution. Requires gene_pools.txt (save opponent genes), test_pools.txt ( save test genes)
genetic_base.py: Class of genetic algorithm
fitness_function.py: Generate fitness function 
vae.py: Variational Autoencoder class. Training. 
test.py: Function to test the algorithm
utility.py: useful functions


3. final_improvement:
improvement.py: Search neighbor (1 gene different 150 neighbors or 2 gene different 11025 neighbors) + crossover to do the final improvement/polish on the gene to improve some performances. 

