import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
import numpy as np
import math,time
from collections import Counter

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

    def __init__(self, x_orig, y_orig, hidden_layer=None, model=None, input_shape=None, device=torch.device("cuda"),
                 n_class=10, n_neighbors=10, query_class="l2", norm_order=2, normalize=False,
                 avg_channel=False, fitness_function=neg_l2_dist, sampling_temperature=.3,
                 max_generation=30, requires_init=False, mut_range=(.05, .15), mut_prob=(.05, .15), mut_mode="crossover",
                 decay=(.9, .9), n_population=1000, memory_threshold=.025, plasma_threshold=.005, return_log=True):

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
        self.n_plasma = int(plasma_threshold*self.n_population)
        self.n_memory = int(memory_threshold*self.n_population)-self.n_plasma
        self.return_log = return_log

        self.model.to(self.device)
        self.model.eval()

        self._query_objects = self._build_all_query_objects()

    def _build_class_query_object(self, xh_orig, class_label=-1):
        if class_label + 1:
            x_class = xh_orig[self.y_orig == class_label]
        else:
            x_class = xh_orig
        if self.query_class == "l2":
            query_object = L2NearestNeighbors(n_neighbors=self.n_neighbors,n_jobs=-1).fit(x_class)
        return query_object

    def _build_all_query_objects(self):
        xh_orig = self._hidden_repr_mapping(self.x_orig)
        if self.n_class:
            print("Building query objects for {} classes {} samples...".format(self.n_class, self.x_orig.size(0)),
                  end="")
            query_objects = [self._build_class_query_object(xh_orig,class_label=i) for i in range(self.n_class)]
            print("done!")
        else:
            print("Building one single query object {} samples...".format(self.x_orig.size(0)), end="")
            query_objects = [self._build_class_query_object(xh_orig)]
            print("done!")
        return query_objects

    def _query_nns_ind(self, Q):
        assert Q.ndim == 2, "Q: 2d array-like (n_queries,n_features)"
        if self.n_class:
            print("Searching {} naive B cells per class for each of {} antigens...".format(self.n_neighbors, len(Q)),
                  end="")
            rel_ind = [query_obj(Q) for query_obj in self._query_objects]
            abs_ind = []
            for c in range(self.n_class):
                class_ind = np.where(self.y_orig.numpy() == c)[0]
                abs_ind.append(class_ind[rel_ind[c]])
            print("done!")
        else:
            print("Searching {} naive B cells for each of {} antigens...".format(self.n_neighbors, Q.size(0)),
                  end="")
            abs_ind = [query_obj(Q) for query_obj in self._query_objects]
            print('done!')
        return abs_ind

    def _hidden_repr_mapping(self, x, batch_size=256):
        '''
        transform b cells and antigens into inner representations of AISE
        '''
        global conv_outputs
        if self.hidden_layer is not None:
            xhs = []
            for i in range(0,x.size(0),batch_size):
                conv_outputs.clear()
                xx = x[i:i+batch_size]
                with torch.no_grad():
                    self.model(xx.to(self.device))
                    xh = conv_outputs[self.hidden_layer].detach().cpu()
                    if self.avg_channel:
                        xh = xh.sum(dim=1)
                    xh = xh.flatten(start_dim=1)
                    if self.normalize:
                        xh = xh/xh.pow(2).sum(dim=1,keepdim=True).sqrt()
                    xhs.append(xh)
            return torch.cat(xhs)
        else:
            xh = x.flatten(start_dim=1)
            if self.normalize:
                xh = xh/xh.pow(2).sum(dim=1,keepdim=True).sqrt()
            return xh.detach().cpu()

    def generate_b_cells(self, ant, ant_tran, nbc_ind, y_ant=None):
        assert ant_tran.ndim == 2, "ant: 2d tensor (n_antigens,n_features)"
        mem_bcs = []
        pla_bcs = []
        mem_labs = []
        pla_labs = []
        print("Affinity maturation process starts with population of {}...".format(self.n_population))
        ant_logs = []  # store the history dict in terms of metrics for antigens
        for n in range(ant.size(0)):
            # print(torch.cuda.memory_summary())
            genadapt = GenAdapt(self.mut_range[1], self.mut_prob[1], mode=self.mut_mode)
            curr_gen = torch.cat([self.x_orig[ind[n]].flatten(start_dim=1) for ind in nbc_ind])  # naive b cells
            labels = np.concatenate([self.y_orig[ind[n]] for ind in nbc_ind])
            if self.requires_init:
                assert self.n_population % (
                        self.n_class * self.n_neighbors) == 0, \
                    "n_population should be divisible by the product of n_class and n_neighbors"
                curr_gen = curr_gen.repeat((self.n_population // (self.n_class * self.n_neighbors), 1))
                curr_gen = genadapt.mutate_random(curr_gen)  # initialize *NOTE: torch.Tensor.repeat <> numpy.repeat
                labels = np.tile(labels, self.n_population // (self.n_class * self.n_neighbors))

            static_index = np.arange(len(self.max_generation))  # static generation indices
            curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
            fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_repr)[0])
            best_pop_fitness = float('-inf')
            decay_coef = (1., 1.)
            num_plateau = 0
            ant_log = dict()  # history log for each antigen
            # zeroth generation logging
            fitness_pop_hist = []
            pop_fitness = fitness_score.sum().item()
            fitness_pop_hist.append(pop_fitness)
            if y_ant is not None:
                fitness_true_class_hist = []
                pct_true_class_hist = []
                true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                fitness_true_class_hist.append(true_class_fitness)
                true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                pct_true_class_hist.append(true_class_pct)

            for i in range(self.max_generation):
                survival_prob = F.softmax(fitness_score / self.sampl_temp, dim=-1)
                parents_ind1 = np.array(Categorical(probs=survival_prob).sample((self.n_population,)))
                if self.mut_mode == "crossover":
                    parents_ind2 = np.concatenate([Categorical(probs=F.softmax(fitness_score[labels==labels[ind]] / self.sampl_temp,
                                                          dim=-1)).sample((1,)) for ind in parents_ind1])
                    parents_ind2 = [static_index[labels==labels[ind1]][ind2] for ind1,ind2 in zip(parents_ind1,parents_ind2)]
                    parents = [curr_gen[parents_ind1],curr_gen[parents_ind2]]
                    curr_gen = genadapt(parents, fitness_score[parents_ind1] /\
                                      (fitness_score[parents_ind1]+fitness_score[parents_ind2]))
                else:
                    parents = curr_gen[parents_ind1]
                    curr_gen = genadapt(parents)
                curr_repr = self._hidden_repr_mapping(curr_gen.view((-1,) + self.x_orig.size()[1:]))
                labels = labels[parents_ind1]
                fitness_score = torch.Tensor(self.fitness_func(ant_tran[n].unsqueeze(0), curr_repr)[0])
                pop_fitness = fitness_score.sum().item()

                # logging
                fitness_pop_hist.append(pop_fitness)
                if y_ant is not None:
                    true_class_fitness = fitness_score[labels == y_ant[n]].sum().item()
                    fitness_true_class_hist.append(true_class_fitness)
                    true_class_pct = (labels == y_ant[n]).astype('float').mean().item()
                    pct_true_class_hist.append(true_class_pct)

                # check homogeneity
                if len(np.unique(labels)) == 1:
                    break # early stop

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
            pla_bcs.append(curr_gen[fitness_rank[-self.n_plasma:]])
            pla_labs.append(labels[fitness_rank[-self.n_plasma:]])
            mem_bcs.append(curr_gen[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]])
            mem_labs.append(labels[fitness_rank[-(self.n_memory+self.n_plasma):-self.n_plasma]])
            ant_logs.append(ant_log)
        print("Memory & plasma B cells generated!")
        return torch.stack(mem_bcs).view((-1,self.n_memory)+self.input_shape), np.stack(mem_labs), \
               torch.stack(pla_bcs).view((-1,self.n_plasma)+self.input_shape), np.stack(pla_labs), \
               ant_logs

    def clonal_expansion(self, ant, y_ant=None):
        print("Clonal expansion starts...")
        ant_tran = self._hidden_repr_mapping(ant.detach())
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
        return AISE.predict_proba(*args).argmax(axis=1)

    @staticmethod
    def predict_proba(*args,n_class):
        assert args
        if len(args) == 1:
            pla_labs = args[0]
        else:
            pla_labs = args[3]
        return np.stack(list(map(lambda x: np.bincount(x,minlength=n_class)/len(x), pla_labs)))

    def __call__(self, ant, y_ant=None):
        return self.clonal_expansion(ant.detach(), y_ant)

if __name__ == "__main__":
    import os,time,pickle
    from datetime import datetime
    from torchvision import transforms, datasets
    from attack import *
    from mnist_model import *
    from sklearn.neighbors import KNeighborsClassifier
    from collections import deque

    import argparse
    parser = argparse.ArgumentParser("AISE Launcher")
    parser.add_argument("--class-num",help="Number of classes",type=int,default=10)
    parser.add_argument("--train-size",help="Training size",type=int,default=2000)
    parser.add_argument("--eval-size",help="Evaluation size",type=int,default=200)
    parser.add_argument("--n-neighbors",help="Number of ancestors for each class",type=int,default=10)
    parser.add_argument("--max-generation",help="Max number of generations",type=int,default=50)
    parser.add_argument("--hidden-layer",help="Specify a hidden layer",type=int)
    parser.add_argument("--sampling-temp",help="Sampling temperature",type=float,default=0.3)
    parser.add_argument("--avg-channel",help="Whether to average the channels or not",action="store_true")
    parser.add_argument("--device", help="CPU/GPU device")
    parser.add_argument("-c","--use-cache",help="Whether cache is used",action="store_true")
    parser.add_argument("-s","--save-result",help="Whether to save the result or not",action="store_true")
    parser.add_argument("-n","--normalize",help="Whether to normalize the flattened vector or not",action="store_true")
    parser.add_argument("-a","--attack",help="Whether to use PGD attacks",action="store_true")

    args = parser.parse_args()

    ROOT = "./datasets"
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.0,std=1.0)
    ])

    N_CLASS = args.class_num
    N_TRAIN = args.train_size
    N_EVAL = args.eval_size
    N_NEIGH = args.n_neighbors
    MAX_GEN = args.max_generation
    HIDDEN_LAYER = args.hidden_layer
    SAMPL_TEMP = args.sampling_temp
    AVG_CHANNEL = args.avg_channel
    if args.device:
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    USE_CACHE = args.use_cache
    SAVE_RESULT = args.save_result
    ATTACK = args.attack
    NORMALIZE = args.normalize
    LAYER_NAME = "conv" + str(HIDDEN_LAYER + 1) if HIDDEN_LAYER is not None else "input"
    DATA_TYPE = "adversarial" if ATTACK else "legitimate"
    DATA_TYPE_SHORT = "adv" if ATTACK else "clean"

    net = CNN().to(DEVICE)
    net.load_state_dict(torch.load("./models/mnistmodel.pt",map_location=DEVICE)["state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    trainset = datasets.MNIST(root=ROOT,train=True,transform=TRANSFORM,download=False)
    testset = datasets.MNIST(root=ROOT,train=False,transform=TRANSFORM,download=False)

    np.random.seed(1234)
    ind_train = np.arange(len(trainset))
    np.random.shuffle(ind_train)
    ind_train = ind_train[:N_TRAIN]

    ind_eval = np.arange(len(testset))
    np.random.shuffle(ind_eval)
    ind_eval = ind_eval[:N_EVAL]

    x_train = trainset.data[ind_train].unsqueeze(1)/255.
    y_train = trainset.targets[ind_train]
    x_eval = testset.data[ind_eval].unsqueeze(1)/255.
    y_eval = testset.targets[ind_eval]

    if ATTACK:
        if USE_CACHE:
            if os.path.exists("./cache/x_v2_{}_{}.pkl".format(DATA_TYPE_SHORT,N_EVAL)):
                with open("./cache/x_v2_{}_{}.pkl".format(DATA_TYPE_SHORT,N_EVAL),"rb") as f:
                    x_adv = torch.Tensor(pickle.load(f)).to(DEVICE)
            else:
                x_adv = PGD(eps=40/255.,sigma=20/255.,nb_iter=20,
                            DEVICE=DEVICE).attack(net,x_eval.to(DEVICE),y_eval.to(DEVICE)).detach().to(DEVICE)
                with open("./cache/x_v2_{}_{}.pkl".format(DATA_TYPE_SHORT,N_EVAL), "wb") as f:
                    pickle.dump(x_adv.detach().cpu().numpy(), f)
        else:
            x_adv = PGD(eps=40/255., sigma=20/255., nb_iter=20,
                        DEVICE=DEVICE).attack(net, x_eval.to(DEVICE), y_eval.to(DEVICE)).detach().to(DEVICE)
        x_ant = x_adv
    else:
        x_ant = x_eval.to(DEVICE)

    *_, out = net(x_ant)
    y_pred = torch.max(out, 1)[1]

    if ATTACK:
        print('The accuracy of plain cnn under PGD attacks is: {}'.format(
            (y_eval.numpy() == y_pred.detach().cpu().numpy()).astype("float").mean()))
    else:
        print("The accuracy of plain cnn on clean data is: {}".format(
            (y_eval.numpy() == y_pred.detach().cpu().numpy()).astype("float").mean()))

    # this is a hook function to register in the model forward
    # register after iterative attacks
    def conv_hook(self, input, output):
        conv_outputs.append(F.relu(output.detach()))
        return None

    for layer in net.modules():
        if layer.__class__.__name__ == "Conv2d":
            layer.register_forward_hook(conv_hook)

    conv_outputs = deque(maxlen=4) # this is a global variable!

    start_time = time.time()
    aise = AISE(x_train,y_train,model=net,n_neighbors=N_NEIGH,hidden_layer=HIDDEN_LAYER,max_generation=MAX_GEN,normalize=NORMALIZE,
                avg_channel=AVG_CHANNEL,sampling_temperature=SAMPL_TEMP)
    mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = aise(x_ant,y_eval)
    end_time = time.time()
    print("Total running time is {}".format(end_time-start_time))
    aise_proba = AISE.predict_proba(pla_labs,n_class=N_CLASS)
    aise_pred = aise_proba.argmax(axis=1)
    aise_acc = (aise_pred==y_eval.numpy()).astype("float").mean()
    print("The accuracy by AISE on {} layer of {} examples is {}".format(LAYER_NAME,DATA_TYPE,aise_acc))

    conv_outputs.clear() # unnecessary
    net(x_train.to(DEVICE))
    train_convs = []
    train_convs.extend(conv_outputs)

    conv_outputs.clear() # unnecessary
    net(x_ant.to(DEVICE))
    ant_convs = []
    ant_convs.extend(conv_outputs)

    if NORMALIZE:
        x_train.div_(x_train.pow(2).sum(dim=(1,2,3),keepdim=True).sqrt())
        train_convs = [train_conv/train_conv.pow(2).sum(dim=(1,2,3),keepdim=True).sqrt() for train_conv in train_convs]
        x_ant.div_(x_ant.pow(2).sum(dim=(1,2,3),keepdim=True).sqrt())
        ant_convs = [ant_conv/ant_conv.pow(2).sum(dim=(1,2,3),keepdim=True).sqrt() for ant_conv in ant_convs]

    knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    knn.fit(train_convs[HIDDEN_LAYER].detach().cpu().flatten(start_dim=1).numpy()
            if HIDDEN_LAYER is not None else x_train.flatten(start_dim=1).numpy(), y_train.numpy())
    knn_proba = knn.predict_proba(ant_convs[HIDDEN_LAYER].detach().cpu().flatten(
        start_dim=1).numpy() if HIDDEN_LAYER is not None
                           else x_ant.flatten(start_dim=1).detach().cpu().numpy())
    knn_pred = knn_proba.argmax(axis=1)
    knn_acc = (knn_pred == y_eval.numpy()).astype("float").mean()
    print("The accuracy by KNN on {} layer of {} examples is {}".format(LAYER_NAME,DATA_TYPE,knn_acc))

    if SAVE_RESULT:
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
        if not os.path.exists("./results"):
            os.mkdir("./results")
        with open("results/result_v2_{}_{}_{}_{}.pkl".format(DATA_TYPE_SHORT,LAYER_NAME,N_EVAL,timestamp),"wb") as f:
            pickle.dump([aise_proba,knn_proba,ant_logs],f)
        with open("results/bcells_v2_{}_{}_{}_{}.pkl".format(DATA_TYPE_SHORT, LAYER_NAME, N_EVAL,timestamp), "wb") as f:
            pickle.dump([mem_bcs.detach().cpu().numpy(),pla_bcs.detach().cpu().numpy()],f)
    else:
        print("The result is:")
        print("AISE:",aise_pred,"KNN",knn_pred)
        print(args)