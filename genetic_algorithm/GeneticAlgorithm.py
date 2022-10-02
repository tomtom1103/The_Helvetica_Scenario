# %%
import numpy as np
import pandas as pd
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tqdm.notebook import tqdm
# %%
class GeneticAlgorithm:
    def __init__(self, df, pop_size=10, num_generations=10, mutation_rate=0.01, crossover_rate=0.5):
        self.df = pd.read_csv(df)
        bins = (2, 5.5, 8)
        group_names = ['bad', 'good']
        target_var = 'quality'
        self.df[target_var] = pd.cut(self.df[target_var], bins = bins, labels = group_names)
        label_quality = LabelEncoder()
        self.df[target_var] = label_quality.fit_transform(self.df[target_var])
        self.data = self.df.drop(target_var, axis=1)
        self.target = self.df[target_var]
        self.var_names = self.data.columns
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_chromosome = []
        self.best_chromosome_score = []
   
    def __repr__(self):
        return f'GeneticAlgorithm(pop_size={self.pop_size}, num_generations={self.num_generations}, mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate})'

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def initialization(self):
        for i in range(self.pop_size):
            self.population.append(np.random.randint(2, size=len(self.var_names)).astype(bool))
        return self.population
        
    def fitness_evaluation(self):
        if self.population:
            pass
        else:
            print('Initializing the first population..')
            self.population = self.initialization()
        
        acc_score = []
        for mask in tqdm(self.population, desc='Calculating Fitness Score..'):
            train_data = self.data[np.array(self.var_names)[mask]]
            x_train, x_test, y_train, y_test = train_test_split(train_data, self.target, test_size=0.2, random_state=0)
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.fit_transform(x_test)
            rfc = RandomForestClassifier(n_estimators=200)
            rfc.fit(x_train, y_train)
            pred_rfc = rfc.predict(x_test)
            acc = accuracy_score(y_test, pred_rfc)
            acc_score.append(acc)
        fitness_dict = {}
        count = 0
        for score in acc_score:
            fitness_dict[count] = score
            count += 1

        self.fitness_dict = fitness_dict
        self.best_chromosome.append(self.population[max(self.fitness_dict, key=self.fitness_dict.get)])
        self.best_chromosome_score.append(max(self.fitness_dict.values()))
        
        print(f'Best chromosome score: {self.best_chromosome_score[-1]}')
        return self.fitness_dict

    def probabilistic_selection(self):
        if self.fitness_dict:
            pass
        else:
            self.fitness_dict = self.fitness_evaluation()
        fitness_score = list(self.fitness_dict.values())
        fitness_score = self.softmax(fitness_score)
        selection = np.random.choice(list(self.fitness_dict.keys()), self.pop_size, p=list(fitness_score), replace=True)
        parent_population = []
        elite = max(self.fitness_dict)
        parent_population.append(self.population[elite])
        for choice in selection[:-1]:
            parent_population.append(self.population[choice])
        
        self.parent_population = parent_population
        return self.parent_population

    def crossover(self):
        if self.parent_population:
            pass
        else:
            self.parent_population = self.probabilistic_selection()

        crossover_population = []
        for i in range(0, len(self.parent_population), 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, len(self.var_names) - 1)
                parent1 = self.parent_population[i]
                parent2 = self.parent_population[i + 1]
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                crossover_population.append(child1)
                crossover_population.append(child2)
            else:
                crossover_population.append(self.parent_population[i])
                crossover_population.append(self.parent_population[i + 1])
        
        self.crossover_population = crossover_population
        self.before_mutation = copy.deepcopy(self.crossover_population)
        return self.crossover_population

    def mutation(self):
        if self.crossover_population:
            pass
        else:
            self.crossover_population = self.crossover()

        mutation_population = []
        for i in range(len(self.crossover_population)):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(0, len(self.var_names))
                mutated_child = self.crossover_population[i]
                mutated_child[mutation_point] = not mutated_child[mutation_point]
                mutation_population.append(mutated_child)
            else:
                mutation_population.append(self.crossover_population[i])
        
        self.population = mutation_population
        return self.population

    def run_algorithm(self):
        for i in tqdm(range(self.num_generations)):
            print(f'Generation {i+1}')
            self.fitness_dict = self.fitness_evaluation()
            self.parent_population = self.probabilistic_selection()
            self.crossover_population = self.crossover()
            self.mutation_population = self.mutation()
        
    def plot(self):
        plt.plot(self.best_chromosome_score)
        plt.xlabel('Generation')
        plt.ylabel('Best Chromosome Score')
        plt.show();

data = GeneticAlgorithm('winequality-red.csv', pop_size=50, num_generations=20, mutation_rate=0.01, crossover_rate=0.5)
#data.run_algorithm()

# %%
