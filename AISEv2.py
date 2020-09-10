import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
import numpy as np
import math,time
from collections import Counter

# this is a hook function to register in the model forward
def conv_hook(self,input,output):
    conv_outputs.append(output.detach())
    return None

class GenAdapt:
    '''
    core component of AISE B-cell generation
    '''

    def __init__(self, mut_range, mut_prob, mode='random'):
        self.mut_range = mut_range
        self.mut_prob = mut_prob
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
            raise ValueError("Unsupported mutation type!")


class L2NearestNeighbors(NearestNeighbors):
    '''
    compatible query object class for euclidean distance
    '''

    def __call__(self, X):
        return self.kneighbors(X, return_distance=False)

def neg_l2_dist(X, Y):
    return -euclidean_distances(X, Y)

def cosine_similarity(X, Y, normalized = False):
    if not normalized:
        X_nm = normalize(X, axis=1)
        Y_nm = normalize(X, axis=1)
    else:
        X_nm,Y_nm = X,Y
    return 1-.5*euclidean_distances(X_nm,Y_nm)

class AISE:
    '''
    implement the Adaptive Immune System Emulation
    '''

    def __init__(self, x_orig, y_orig, hidden_layer=None, model=None, input_shape=None,
                 device=torch.device("cuda"), n_class=10, n_neighbors=10, query_class="l2", norm_order=2,
                 normalize=False, avg_channel=False, fitness_function=neg_l2_dist, sampling_temperature=.3,
                 max_generation=20, requires_init=False,
                 mut_range=(.1, .3), mut_prob=(.1, .3), mut_mode="crossover",
                 decay=(.9, .9), n_population=1000, memory_threshold=.25, plasma_threshold=.05, return_log=True):

        self.model = model
        self.device = device

        if input_shape is None:
            self.input_shape = tuple(x_orig.shape[1:])  # mnist: (1,28,28)
        else:
            self.input_shape = input_shape

        self.x_orig = x_orig
        self.y_orig = y_orig

        self.hidden_layer = hidden_layer

        self.n_class = n_class
        self.n_neighbors = n_neighbors
        self.query_class = query_class
        self.norm_order = norm_order
        self.normalize = normalize
        self.avg_channel = avg_channel
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
        self.decay = decay
        self.n_population = n_population
        self.plasma_thres = plasma_threshold
        self.memory_thres = memory_threshold
        self.return_log = return_log

        self.model.to(self.device)
        self.model.eval()
        
        self.xh = self._transform_to_inner_repr(self.x_orig)
        self.query_objects = self._build_all_query_objects()

    def _build_class_query_object(self, class_label=-1):
        if class_label + 1:
            x_class = self.xh[self.y_orig == class_label]
        else:
            x_class = self.xh
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors).fit(x_class)
        return query_object

    def _build_all_query_objects(self):
        if self.n_class:
            print("Building query objects for {} classes {} samples...".format(self.n_class, self.x_orig.size(0)),
                  end="")
            query_objects = [self._build_class_query_object(i) for i in range(self.n_class)]
            print("done!")
        else:
            print("Building one single query object {} samples...".format(self.x_orig.size(0)), end="")
            query_objects = [self._build_class_query_object()]
            print("done!")
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            print("Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, len(Q)),
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

    def _transform_to_inner_repr(self, x, batch_size=256):
        '''
        transform b cells and antigens into inner representations of AISE
        '''
        global conv_outputs
        if self.hidden_layer:
            xs = []
            for i in range(0,x.size(0),batch_size):
                conv_outputs = []
                xx = x[i*batch_size:(i+1)*batch_size]
                with torch.no_grad():
                    self.model(xx.to(self.device))
                    xh = conv_outputs[self.hidden_layer].detach().cpu()
                    if self.avg_channel:
                        xh = xh.sum(dim=1)
                    xh = xh.flatten(start_dim=1)
                    if self.normalize:
                        xh = xh/xh.pow(2).sum(dim=1,keepdim=True).sqrt()
                    xs.append(xh)
                return torch.cat(xs)
        else:
            return x.flatten(start_dim=1).cpu()

    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        mem_bc_batch = []
        pla_bc_batch = []
        mem_lab_batch = []
        pla_lab_batch = []
        print("Affinity maturation process starts with population of {}...".format(self.n_population))
        ant_logs = []  # store the history dict in terms of metrics for antigens
        for n in range(ant.size(0)):
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], mode=self.mut_mode)
            curr_gen = torch.cat([self.x_orig[ind[n]].flatten(start_dim=1) for ind in nbc_ind])  # naive b cells
            labels = np.concatenate([self.y_orig[ind[n]] for ind in nbc_ind])
            static_index = np.arange(len(labels))
            if self.requires_init:
                assert self.n_population % (
                        self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = np.tile(labels, self.n_population // (self.n_class * self.n_neighbors))
            curr_inner_repr = self._transform_to_inner_repr(curr_gen.view((-1,)+self.x_orig.size()[1:]))
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
                    curr_gen = genadapt(parents)
                curr_inner_repr = self._transform_to_inner_repr(curr_gen.view((-1,)+self.x_orig.size()[1:]))
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
                                            mode=self.mut_mode)
                    else:
                        best_pop_fitness = pop_fitness
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
        return torch.cat(mem_bc_batch), np.stack(mem_lab_batch), \
               torch.cat(pla_bc_batch), np.stack(pla_lab_batch), \
               ant_logs

    def clonal_expansion(self, ant, y_ant=None):
        print("Clonal expansion starts...")
        ant_tran = self._transform_to_inner_repr(ant.detach())
        nbc_ind = self._query_nns_ind(ant_tran.numpy())
        mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = self.generate_b_cells(ant.flatten(start_dim=1), ant_tran,
                                                                               nbc_ind, np.array(y_ant))
        print("{} plasma B cells and {} memory generated!".format(pla_bcs.size(0), mem_bcs.size(0)))
        if self.return_log:
            return mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs
        else:
            return mem_bcs, mem_labs, pla_bcs, pla_labs

    @staticmethod
    def predict(*args):
        assert args
        if len(args) == 1:
            pla_labs = args[0]
        else:
            pla_labs = args[3]
        return np.array(list(map(lambda x: Counter(np.array(x)).most_common(1)[0][0], pla_labs)))

    def __call__(self, ant, y_ant=None):
        return self.clonal_expansion(ant, y_ant)

if __name__ == "__main__":
    import os,time,pickle
    from torchvision import transforms, datasets
    from attack import *
    from mnist_model import *
    from sklearn.neighbors import KNeighborsClassifier

    ROOT = "./datasets"
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.0,std=1.0)
    ])
    N_TRAIN = 2000
    N_EVAL = 200

    DEVICE = torch.device("cuda")

    net = CNN().to(DEVICE)
    net.load_state_dict(torch.load("./models/mnistmodel.pt",map_location=DEVICE)["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for layer in net.modules():
        if layer.__class__.__name__ == "Conv2d":
            layer.register_forward_hook(conv_hook)

    trainset = datasets.MNIST(root=ROOT,train=True,transform=TRANSFORM,download=False)
    testset = datasets.MNIST(root=ROOT,train=False,transform=TRANSFORM,download=False)
    np.random.seed(1234)
    ind_full = np.arange(len(trainset))
    np.random.shuffle(ind_full)
    ind_train = ind_full[:N_TRAIN]
    ind_eval = ind_full[N_TRAIN:N_TRAIN+N_EVAL]
    x_train = trainset.data[ind_train].unsqueeze(1)/255.
    y_train = trainset.targets[ind_train]
    x_eval = trainset.data[ind_eval].unsqueeze(1)/255.
    y_eval = trainset.targets[ind_eval]

    conv_outputs = []
    if os.path.exists("./cache/x_adv_test.pkl"):
        with open("./cache/x_adv_test.pkl","rb") as f:
            x_adv = torch.Tensor(pickle.load(f)).to(DEVICE)
    else:
        x_adv = PGD(eps=40/255.,sigma=20/255.,nb_iter=20,
                    DEVICE=DEVICE).attack(net,x_eval.to(DEVICE),y_eval.to(DEVICE))
        with open("./cache/x_adv_test.pkl","wb") as f:
            pickle.dump(x_adv.detach().cpu().numpy(),f)
    *_, out = net(x_adv)
    y_pred_adv = torch.max(out, 1)[1]
    print('The accuracy of plain cnn under PGD attacks is: {:f}'.format(
        (y_eval.numpy() == y_pred_adv.detach().cpu().numpy()).astype("float").mean()))

    start_time = time.time()
    aise = AISE(x_train,y_train,model=net)
    mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = aise(x_adv,y_eval)
    end_time = time.time()
    print("Total running time is {}".format(end_time-start_time))
    aise_pred = AISE.predict(pla_labs)
    aise_acc = (aise_pred==y_eval.numpy()).astype("float").mean()
    print("The accuracy by AISE on adversarial examples is {}".format(aise_acc))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train.flatten(start_dim=1).numpy(),y_train.numpy())
    knn_pred = knn.predict(x_adv.flatten(start_dim=1).detach().cpu().numpy())
    knn_acc = (knn_pred == y_eval.numpy()).astype("float").mean()
    print("The accuracy by KNN on adversarial examples is {}".format(knn_acc))

    if not os.path.exists("./results"):
        os.mkdir("./results")
    with open("results/result_200_v2.pkl","wb") as f:
        pickle.dump(ant_logs,f)