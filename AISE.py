import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math


class GenAdapt:
    '''
    core component of AISE B-cell generation
    '''

    def __init__(self, mut_range, mut_prob, combine_prob=0, mode='random'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.combine_prob = combine_prob
        self.mode = mode

    def crossover(self, p1, p2, select_prob):
        assert p1.ndim == 1 and p2.ndim == 1, "Number of dimensions should be 1"
        crossover_mask = np.random.random(p1.size(0)) < select_prob
        return torch.where(crossover_mask, p1, p2)

    def mutate_random(self, parent):
        mut = 2 * torch.rand_like(parent) - 1  # uniform (-1,1)
        mut = self.mut_range * mut
        mut_mask = torch.rand(parent.size()) < self.mut_prob
        child = torch.where(mut_mask, parent, parent + mut)
        return torch.clamp(child, 0, 1)

    def mutate_guided(self, parent, guidance):
        mut = (2 * torch.rand_like(parent) - 1) * guidance * self.mut_range  # uniform (-1,1)
        mut_mask = torch.rand(parent.size()) < self.mut_prob
        child = torch.where(mut_mask, parent, parent + mut)
        return torch.clamp(child, 0, 1)

    def mutate_combined(self, parent, guidance):
        mut = 2 * torch.rand_like(parent) - 1  # uniform (-1,1)
        # when self.combine_prob is set 0, it degenerate into random mutate
        combine_mask = torch.rand(parent.size()) < self.combine_prob
        mut = self.mut_range * torch.where(combine_mask, guidance, mut)
        mut_mask = torch.rand(parent.size()) < self.mut_prob
        child = torch.where(mut_mask, parent, parent + mut)
        return torch.clamp(child, 0, 1)

    def __call__(self, parent, guidance=None):
        if self.mode == "random":
            return self.mutate_random(parent)
        if self.mode == "guided":
            assert guidance is not None, 'guided mutation must have guidance'
            return self.mutate_guided(parent, guidance)
        if self.mode == "combined":
            assert guidance is not None, 'combined mutation should have guidance'
            return self.mutate_combined(parent, guidance)
        else:
            raise ValueError("Unsupported mutation type!")

    def proliferate(self, p1, p2, select_prob, mut_prob):
        pass


class L2NearestNeighbors(NearestNeighbors):
    '''
    compatible query object class for euclidean distance
    '''

    def __call__(self, X):
        return self.kneighbors(X, return_distance=False)


def recip_l2_dist(X, Y, eps=1e-6):
    correction = np.power(euclidean_distances(X, Y), 2) + eps
    return 1 / np.sqrt(correction)


def neg_l2_dist(X, Y):
    return -euclidean_distances(X, Y)


class AISE:
    '''
    implement the Adaptive Immune System Emulation
    '''

    def __init__(self, X_orig, y_orig, X_hidden=[], layer_dims=None, model=None, input_shape=None,
                 device=torch.device("cuda"),
                 n_class=10, n_neighbors=10, query_class="l2", norm_order=2, fitness_function=recip_l2_dist,
                 sampling_temperature=.3, max_generation=20, requires_init=False, mut_range=(.1, .3),
                 mut_prob=(.1, .3),
                 mut_mode="guided", combine_prob=0, decay=(.9,.9), n_population=1000, memory_threshold=.25,
                 plasma_threshold=.05):

        self.model = model
        self.device = device

        if input_shape is None:
            self.input_shape = tuple(X_orig.shape[1:])  # mnist: (1,28,28)
        else:
            self.input_shape = input_shape

        self.X_orig = X_orig.flatten(start_dim=1)
        self.y_orig = y_orig

        self.layer_dims = layer_dims
        self.transform = lambda x, *y: x

        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self.query_class = query_class
        self.norm_order = norm_order
        self.fitness_func = fitness_function
        self.sampl_temp = sampling_temperature
        self.max_generation = max_generation
        self.n_population = self.n_class * self.n_neighbors
        self.requires_init = requires_init

        self.mut_range = mut_range
        self.mut_prob = mut_prob

        if isinstance(mut_range, float):
            self.mut_range = (mut_range, mut_range)
        if isinstance(mut_prob, float):
            self.mut_prob = (mut_prob, mut_prob)

        self.mut_mode = mut_mode
        self.combine_prob = combine_prob
        self.decay = decay
        self.n_population = n_population
        self.plasma_thres = plasma_threshold
        self.memory_thres = memory_threshold

        model.to(device)

        if len(X_hidden) != 0:
            print("Hidden representation found!")
            self.X_hidden = [Xh.flatten(start_dim=1) for Xh in X_hidden]
            if self.layer_dims is None:  # override the self.n_layers
                self.layer_dims = range(len(X_hidden) // 2)  # pick the shallow half of hidden layers
            self.mean_norms = self._calc_mean_norms()
            self.transform = self._get_transform()
            print("Concatenating the input and hidden representations...")
        else:
            self.X_hidden = X_hidden
            self.layer_dims = []

        self.X_cat = self.transform(self.X_orig, *self.X_hidden)
        self.query_objects = self._build_all_query_objects()

    def _calc_mean_norms(self):
        out = []
        assert self.X_orig.ndim == 2
        # out.append(torch.norm(self.X_orig,p=self.norm_order,dim=-1).mean().item())
        for i in self.layer_dims:
            Xh = self.X_hidden[i]
            assert Xh.ndim == 2
            out.append(torch.norm(Xh, p=self.norm_order, dim=-1).mean().item())
        return np.array(out)

    def _get_transform(self):
        weights = 1 / self.mean_norms
        weights = weights / np.sum(weights)

        def transform(x_orig, *args):
            x_cat = []
            # x_cat.append(weights[0]*x_orig)
            for w, arg in zip(weights[1:], args):
                x_cat.append(w * arg)
            return torch.cat(x_cat, dim=1)

        return transform

    def _build_class_query_object(self, class_label=-1):
        if class_label + 1:
            X_class = self.X_cat[self.y_orig == class_label]
        else:
            X_class = self.X_cat
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors).fit(X_class)
        return query_object

    def _build_all_query_objects(self):
        if self.n_class:
            print("Building query objects for {} classes {} samples...".format(self.n_class, self.X_orig.size(0)),
                  end="")
            query_objects = [self._build_class_query_object(i) for i in range(self.n_class)]
            print("done!")
        else:
            print("Building one single query object {} samples...".format(self.X_orig.size(0)), end="")
            query_objects = [self._build_class_query_object()]
            print("done!")
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            print("Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, Q.size(0)),
                  end="")
            rel_ind = [query_obj(Q) for query_obj in self.query_objects]
            abs_ind = []
            for c in range(self.n_class):
                class_ind = np.where(self.y_orig.numpy() == c)[0]
                abs_ind.append(class_ind[rel_ind[c]])
            print("done!")
        else:
            print("Searching {} naive B cells for each of {} antigens...".format(self.n_neighbors, Q.size(0)),
                  end="")
            abs_ind = [query_obj(Q) for query_obj in self.query_objects]
            print('done!')
        return abs_ind

    def _transform_to_inner_repr(self, X, reshape=True):
        '''
        transform b cells and antigens into inner representations of AISE
        '''
        if reshape:
            inputs = X.view((-1,) + self.input_shape)
        else:
            inputs = X
        X_hidden = []
        *out_hidden, _ = self.model(inputs.to(self.device))
        for i in self.layer_dims:
            X_hidden.append(out_hidden[i].detach().cpu().flatten(start_dim=1))
        return self.transform(X.flatten(start_dim=1), *X_hidden)

    def generate_b_cells(self, ant_tran, nbc_ind):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        mem_bc_batch = []
        pla_bc_batch = []
        mem_lab_batch = []
        pla_lab_batch = []
        print("Affinity maturation process starts with population of {}...".format(self.n_population))
        total_fitness_log = []
        for n in range(ant_tran.size(0)):
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], self.combine_prob, mode=self.mut_mode)
            curr_gen = torch.cat([self.X_orig[ind[n]] for ind in nbc_ind])  # naive b cells
            # labels = np.repeat(np.arange(self.n_class), self.n_neighbors)
            labels = np.concatenate([self.y_orig[ind[n]] for ind in nbc_ind])
            if self.requires_init:
                assert self.n_population % (
                            self.n_class * self.n_neighbors) == 0, "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = np.tile(labels, self.n_population // (self.n_class * self.n_neighbors))
            curr_inner_repr = self._transform_to_inner_repr(curr_gen)
            fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_inner_repr)[0])
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            curr_fitness_hist = []
            for i in range(self.max_generation):
                # print("Antigen {} Generation {}".format(n,i))
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                parents_ind = Categorical(probs=survival_prob).sample((self.n_population,))
                parents = curr_gen[parents_ind]
                curr_gen = genadapt(parents, ant_tran[n].unsqueeze(0) - parents)
                curr_inner_repr = self._transform_to_inner_repr(curr_gen)
                labels = labels[parents_ind.numpy()]
                fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_inner_repr)[0])
                pop_fitness = fitness_score.sum().item()
                curr_fitness_hist.append(pop_fitness)
                if self.decay:
                    assert len(self.decay) == 2
                    if pop_fitness < best_pop_fitness:
                        if num_plateau >= max(math.log(self.mut_range[0] / self.mut_range[1], self.decay[0]),
                                              math.log(self.mut_prob[0] / self.mut_prob[1], self.decay[1])):
                            # early stop
                            break
                        decay_coef = tuple(decay_coef[i] * self.decay[i] for i in range(2))
                        num_plateau += 1
                        genadapt = GenAdapt(max(self.mut_range[0], self.mut_range[1] * decay_coef[0]),
                                            max(self.mut_prob[0], self.mut_prob[1] * decay_coef[1]),
                                            self.combine_prob, mode=self.mut_mode)
                    else:
                        best_pop_fitness = pop_fitness
            # fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0),curr_inner_repr)[0])
            _, fitness_rank = torch.sort(fitness_score)
            total_fitness_log.append(curr_fitness_hist)
            pla_bc_batch.append(curr_gen[fitness_rank[-int(self.plasma_thres * self.n_population):]])
            pla_lab_batch.append(labels[fitness_rank[-int(self.plasma_thres * self.n_population):]])
            mem_bc_batch.append(curr_gen[fitness_rank[-int(self.memory_thres * self.n_population):-int(
                self.plasma_thres * self.n_population)]])
            mem_lab_batch.append(labels[fitness_rank[-int(self.memory_thres * self.n_population):-int(
                self.plasma_thres * self.n_population)]])
        print("Memory & plasma B cells generated!")
        return torch.cat(mem_bc_batch), torch.tensor(np.stack(mem_lab_batch)), \
               torch.cat(pla_bc_batch), torch.tensor(np.stack(pla_lab_batch)), \
               total_fitness_log

    def clonal_expansion(self, ant, return_log=False):
        print("Clonal expansion starts...")
        ant_tran = self._transform_to_inner_repr(ant, reshape=False)
        nbc_ind = self._query_nns_ind(ant_tran)
        mem_bcs, mem_labs, pla_bcs, pla_labs, fit_log = self.generate_b_cells(ant_tran, nbc_ind)
        print("{} plasma B cells and {} memory generated!".format(pla_bcs.size(0), mem_bcs.size(0)))
        if return_log:
            return mem_bcs, mem_labs, pla_bcs, pla_labs, fit_log
        else:
            return mem_bcs, mem_labs, pla_bcs, pla_labs
