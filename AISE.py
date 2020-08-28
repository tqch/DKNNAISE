import torch
# import torch.nn as nn
# import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class GenAdapt:
    '''
    core component of AISE B-cell generation
    '''
    def __init__(self,mut_range,mut_norm=2):
        self.mut_range = mut_range
        self.mut_norm=mut_norm
        # self.fitness_func = fitness_function

    def crossover(self,p1,p2,select_prob):
        assert p1.ndim == 1 and p2.ndim == 1, "Number of dimension should be 1"
        crossover_mask = np.random.random(p1.size(0)) < select_prob
        return torch.where(crossover_mask,p1,p2)

    def child_mut(self,child,mut_prob):
        mut = torch.randn_like(child)
        mut = np.random.sample(1).item()*self.mut_range*mut/torch.norm(mut,p=self.mut_norm)
        mut_mask = torch.rand(child.size(0)) < mut_prob
        child_mut = torch.where(mut_mask, child, child+mut)
        return torch.clamp(child_mut,0,1)

    def proliferate(self,p1,p2,select_prob,mut_prob):
        pass

class L2NearestNeighbors(NearestNeighbors):
    '''
    compatible query object class for euclidean distance
    '''
    def __call__(self,X):
        return self.kneighbors(X,return_distance=False)

class AISE:
    '''
    implement the Adaptive Immune System Emulation
    '''
    def __init__(self,X_orig,y_orig,n_class=10,n_neighbors=10,query_class="l2",
                 fitness_function=euclidean_distances,max_generation=10,n_population=100,mut_range=.05,mut_norm=2,memory_threshold=.25,plasma_threshold=.05):

        # self.model = model.to(device)
        self.X_orig = X_orig
        self.y_orig = y_orig
        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self.query_class = query_class
        self.fitness_func = fitness_function
        self.max_generation = max_generation
        self.n_population = n_population
        self.mut_range = mut_range
        self.mut_norm = mut_norm
        # self.n_population = n_population
        self.plasma_thres = plasma_threshold
        self.memory_thres = memory_threshold
        # self.device = device

        self.query_objects = self._build_all_query_objects()


    def _build_class_query_object(self,class_label):
        X_class = self.X_orig[self.y_orig==class_label]
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors).fit(X_class)
        return query_object

    def _build_all_query_objects(self):
        print("Building query objects...",end="")
        query_objects = [self._build_class_query_object(i) for i in range(self.n_class)]
        print("done!")
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        return [query_obj(Q) for query_obj in self.query_objects]

    def generate_b_cells(self, ant, nbc_ind):
        assert ant.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        genadapt = GenAdapt(self.mut_range,self.mut_norm)
        labels_batch = []
        bc_batch = []
        print("Affinity maturation process starts...")
        for n in range(ant.size(0)):
            curr_gen = torch.cat([self.X_orig[ind[n]] for ind in nbc_ind]) # naive b cells
            labels = np.repeat(np.arange(self.n_class),self.n_neighbors)
            for _ in range(self.max_generation):
                fitness_score = self.fitness_func(ant[0].unsqueeze(0),curr_gen)[0]
                unfitted = np.argmax(fitness_score) # as opposed to the definition, not min
                fitted_mask = np.arange(len(labels)) != unfitted
                labels = labels[fitted_mask]
                curr_gen = curr_gen[fitted_mask]
                curr_gen = torch.stack([genadapt.child_mut(bc,mut_prob=.5) for bc in curr_gen])
            fitness_score = self.fitness_func(ant[0].unsqueeze(0),curr_gen)[0]
            fitness_rank = np.argsort(fitness_score)
            labels_batch.append(labels[fitness_rank[:int(self.memory_thres*labels.size)]])
            bc_batch.append(curr_gen[fitness_rank[:int(self.memory_thres*labels.size)]])
        print("Memory & plasma B-cells generated!")
        return torch.tensor(np.stack(labels_batch)),torch.cat(bc_batch)

    def clonal_expansion(self, ant):
        nbc_ind = self._query_nns_ind(ant)
        labels,bcs = self.generate_b_cells(ant,nbc_ind)
        return labels,bcs