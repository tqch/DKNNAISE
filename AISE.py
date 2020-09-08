import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
from collections import Counter


class GenAdapt:
    '''
    core component of AISE B-cell generation
    '''

    def __init__(self, mut_range, mut_prob, combine_rate=0.5, hybrid_rate=0.5, mode='random'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
        self.combine_rate = combine_rate
        self.hybrid_rate = hybrid_rate
        self.mode = mode

    def crossover(self, base1, base2, select_prob):
        assert base1.ndim == 2 and base2.ndim == 2, "Number of dimensions should be 2"
        crossover_mask = torch.rand(base1.size()) < torch.Tensor(select_prob[:,None])
        return torch.where(crossover_mask, base1, base2)

    def mutate_random(self, base):
        mut = 2 * torch.rand_like(base) - 1  # uniform (-1,1)
        mut = self.mut_range * mut
        mut_mask = torch.rand(base.size()) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, 1)

    def mutate_guided(self, base, target):
        guidance = target - base
        mut = (2 * torch.rand_like(base) - 1) * guidance * self.mut_range  # uniform (-1,1)
        mut_mask = torch.rand(base.size()) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, 1)

    def mutate_combined(self, base, target):
        guidance = target - base
        mut_random = 2 * torch.rand_like(base) - 1  # uniform (-1,1)
        mut_guided = (2 * torch.rand_like(base) - 1) * guidance
        # when self.combine_rate is set 0, it degenerate into random mutate
        combine_mask = torch.rand(base.size()) < self.combine_rate
        mut = self.mut_range * torch.where(combine_mask, mut_guided, mut_random)
        mut_mask = torch.rand(base.size()) < self.mut_prob
        child = torch.where(mut_mask, base, base + mut)
        return torch.clamp(child, 0, 1)

    def hybrid(self, base, target):
        child = self.crossover(base, target, self.hybrid_rate)
        child = self.mutate_guided(child, target - base)
        return child

    def crossover_complete(self, parents, select_prob):
        parent1,parent2 = parents
        child = self.crossover(parent1,parent2,select_prob)
        child = self.mutate_random(child)
        return child

    def __call__(self, *args):
        if self.mode == "random":
            base, *_ = args
            return self.mutate_random(base)
        elif self.mode == "crossover":
            assert len(args) == 2
            parents, select_prob = args
            return self.crossover_complete(parents,select_prob)
        else:
            assert len(args) == 2
            base, target = args
            if self.mode == "guided":
                return self.mutate_guided(base, target)
            if self.mode == "combined":
                return self.mutate_combined(base, target)
            if self.mode == "hybrid":
                return self.hybrid(base, target)
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

    def __init__(self, X_orig, y_orig, X_hidden=[], layer_dims=[], model=None, input_shape=None,
                 device=torch.device("cuda"), n_class=10, n_neighbors=10, query_class="l2", norm_order=2,
                 fitness_function=neg_l2_dist, sampling_temperature=.3, max_generation=20, requires_init=False,
                 mut_range=(.1, .3), mut_prob=(.1, .3), mut_mode="combined", combine_rate=0.7, hybrid_rate=.9,
                 decay=(.9, .9), n_population=1000, memory_threshold=.25, plasma_threshold=.05, return_log=True):

        self.model = model
        self.device = device

        if input_shape is None:
            self.input_shape = tuple(X_orig.shape[1:])  # mnist: (1,28,28)
        else:
            self.input_shape = input_shape

        self.X_orig = X_orig.flatten(start_dim=1)
        self.y_orig = y_orig

        self.layer_dims = layer_dims

        self.mean_norms = []
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
        self.combine_rate = combine_rate
        self.hybrid_rate = hybrid_rate
        self.decay = decay
        self.n_population = n_population
        self.plasma_thres = plasma_threshold
        self.memory_thres = memory_threshold
        self.return_log = return_log

        model.to(device)

        if self.layer_dims:
            assert X_hidden is not None
        if X_hidden:
            print("Hidden representation found!")
            self.X_hidden = [Xh.flatten(start_dim=1) for Xh in X_hidden]
            if self.layer_dims is None:  # override the self.n_layers
                self.layer_dims = range(len(X_hidden) // 2)  # pick the shallow half of hidden layers
            self.mean_norms = self._calc_mean_norms()
            self.transform = self._get_transform()
            print("Concatenating the hidden representations...")
        else:
            self.X_hidden = []
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
        if len(self.mean_norms):
            weights = 1 / self.mean_norms
            weights = weights / np.sum(weights)
        else:
            weights = self.mean_norms

        def transform(x_orig, *args):
            x_cat = []
            if len(weights):
                for w, arg in zip(weights, args):
                    x_cat.append(w * arg)
            else:
                x_cat.append(x_orig)
            return x_cat[0] if len(x_cat) else torch.cat(x_cat, dim=1)

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

    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        mem_bc_batch = []
        pla_bc_batch = []
        mem_lab_batch = []
        pla_lab_batch = []
        print("Affinity maturation process starts with population of {}...".format(self.n_population))
        ant_logs = []  # store the history dict in terms of metrics for antigens
        for n in range(ant.size(0)):
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], self.combine_rate,
                                self.hybrid_rate, mode=self.mut_mode)
            curr_gen = torch.cat([self.X_orig[ind[n]] for ind in nbc_ind])  # naive b cells
            # labels = np.repeat(np.arange(self.n_class), self.n_neighbors)
            labels = np.concatenate([self.y_orig[ind[n]] for ind in nbc_ind])
            static_index = np.arange(len(labels))
            if self.requires_init:
                assert self.n_population % (
                        self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = np.tile(labels, self.n_population // (self.n_class * self.n_neighbors))
            curr_inner_repr = self._transform_to_inner_repr(curr_gen)
            fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_inner_repr)[0])
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            ant_log = dict()  # history log for each antigen
            fitness_pop_hist = []
            if y_ant is not None:
                fitness_true_class_hist = []
                pct_true_class_hist = []
            for i in range(self.max_generation):
                # print("Antigen {} Generation {}".format(n,i))
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                parents_ind1 = Categorical(probs=survival_prob).sample((self.n_population,))
                if self.mut_mode == "crossover":
                    parents_ind2 = torch.cat([Categorical(probs=F.softmax(fitness_score[labels==labels[ind]] / self.sampl_temp,
                                                            dim=-1)).sample((1,)) for ind in parents_ind1])
                    parents_ind2 = [static_index[labels==labels[ind1]][ind2] for ind1,ind2 in zip(parents_ind1,parents_ind2)]
                    static_index = np.arange(self.n_population)
                    parents = [curr_gen[parents_ind1],curr_gen[parents_ind2]]
                    curr_gen = genadapt(parents, fitness_score[parents_ind1] /\
                                        (fitness_score[parents_ind1]+fitness_score[parents_ind2]))
                else:
                    parents = curr_gen[parents_ind1]
                    curr_gen = genadapt(parents, ant[n].unsqueeze(0))
                curr_inner_repr = self._transform_to_inner_repr(curr_gen)
                labels = labels[parents_ind1.numpy()]
                fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_inner_repr)[0])
                pop_fitness = fitness_score.sum().item()
                # logging
                fitness_pop_hist.append(pop_fitness)
                if y_ant is not None:
                    true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                    fitness_true_class_hist.append(true_class_fitness)
                    true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                    pct_true_class_hist.append(true_class_pct)
                # adaptive shrinkage of certain hyper-parameters
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
                                            self.combine_rate, self.hybrid_rate, mode=self.mut_mode)
                    else:
                        best_pop_fitness = pop_fitness
            # fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0),curr_inner_repr)[0])
            _, fitness_rank = torch.sort(fitness_score)
            ant_log["fitness_pop"] = fitness_pop_hist
            if y_ant is not None:
                ant_log["fitness_true_class"] = fitness_true_class_hist
                ant_log["pct_true_class"] = pct_true_class_hist
            pla_bc_batch.append(curr_gen[fitness_rank[-int(self.plasma_thres * self.n_population):]])
            pla_lab_batch.append(labels[fitness_rank[-int(self.plasma_thres * self.n_population):]])
            mem_bc_batch.append(curr_gen[fitness_rank[-int(self.memory_thres * self.n_population):-int(
                self.plasma_thres * self.n_population)]])
            mem_lab_batch.append(labels[fitness_rank[-int(self.memory_thres * self.n_population):-int(
                self.plasma_thres * self.n_population)]])
            ant_logs.append(ant_log)
        print("Memory & plasma B cells generated!")
        return torch.cat(mem_bc_batch), torch.tensor(np.stack(mem_lab_batch)), \
               torch.cat(pla_bc_batch), torch.tensor(np.stack(pla_lab_batch)), \
               ant_logs

    def clonal_expansion(self, ant, y_ant=None, return_log=False):
        print("Clonal expansion starts...")
        ant_tran = self._transform_to_inner_repr(ant, reshape=False)
        nbc_ind = self._query_nns_ind(ant_tran)
        mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = self.generate_b_cells(ant.flatten(start_dim=1), ant_tran,
                                                                               nbc_ind, y_ant)
        print("{} plasma B cells and {} memory generated!".format(pla_bcs.size(0), mem_bcs.size(0)))
        if return_log:
            return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs
        else:
            return mem_bcs, mem_labs, pla_bcs, pla_labs

    @staticmethod
    def predict(*args):
        assert args
        if len(args) == 1:
            pla_labs = args
        else:
            pla_labs = args[3]
        return np.array(list(map(lambda x: Counter(x).most_common(1)[0][0], pla_labs.numpy())))

    def __call__(self, ant, y_ant=None):
        # check X_cat
        if np.any(self._calc_mean_norms() != self.mean_norms) or \
                len(self._calc_mean_norms()) != len(self.mean_norms):
            self.mean_norms = self._calc_mean_norms()
            self.transform = self._get_transform()
            print("Concatenating the hidden representations...")
            self.X_cat = self.transform(self.X_orig, *self.X_hidden)
            self.query_objects = self._build_all_query_objects()
        # check n_class
        elif len(self.query_objects) != self.n_class:
            self.query_objects = self._build_all_query_objects()

        return self.clonal_expansion(ant, y_ant, self.return_log)