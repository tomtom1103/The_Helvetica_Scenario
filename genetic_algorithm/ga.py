# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
# %%
class GeneticAlgorithm:
    def __init__(self, df, target_var, pop_size=10, num_generations=100, mutation_rate=0.01, crossover_rate=0.5):
        self.df = pd.read_csv(df)
        self.data = self.df.drop({target_var}, axis=1)
        self.target = self.df[target_var]
        self.var_names = self.data.columns
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
   
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
            print('exists')
            pass
        else:
            print('doesnt exist')
            self.population = self.initialization()

        auc_score = []
        population_list = []
        for mask in self.population:
            train_data = self.data[np.array(self.var_names)[mask]]
            x_train, x_test, y_train, y_test = train_test_split(train_data, self.target, test_size=0.2, random_state=0)
            lr = LogisticRegression(penalty='none', max_iter=100)
            lr.fit(x_train, y_train)
            lr_acc = accuracy_score(y_test, lr.predict(x_test))
            auc = roc_auc_score(y_test, lr.predict_proba(x_test), multi_class='ovr')
            auc_score.append(auc)
        fitness_dict = {}
        count = 0
        for score in auc_score:
            fitness_dict[count] = score
            count += 1
        
        return fitness_dict

    def selection(self):
        pass


        



data = GeneticAlgorithm('winequality-red.csv',target_var='quality')


# %%
x = pd.read_csv('winequality-red.csv')
tmp = np.random.randint(2,size=len(data.variables)).astype(bool)
print(tmp)
print(np.array(x.columns)[tmp])
x[np.array(x.columns)[tmp]]

# %%
auc_score = [0.5786054871469283,
    0.6302770594742367,
 0.5087278913967281,
 0.5996988410324334,
 0.6661359995141165,
 0.5446831448281574,
 0.6819356790108934,
 0.7529791445324202,
 0.6017414263355557,
 0.561256739462517]

