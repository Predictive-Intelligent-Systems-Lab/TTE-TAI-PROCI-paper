import math
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from collections import OrderedDict
from typing import Any


class WindowDataset(IterableDataset):
    """
    Yields ONE window example at a time: x[k_end-w : k_end, :]

    - Uses burst sampling for locality (choose a eligible few traj -> sample windows)
    - Uses a small LRU cache of open traj memmaps per worker process
    - If samples_per_epoch is set, yields exactly that many samples total per epoch
      (split across workers if num_workers > 0)

    Note:
        - When creating validation loader with DataLoader, if num_workers is modified, so is the generated dataset, which is usually unwanted. Keep same number of workers for reproducibility.
          This happens because a different seed is assigned to each worker, hence a different random generator (rng) that will randomly sample examples,
        - The size of eligible trajectories should be bigger than the number of workers.
    """
    def __init__(
        self,
        dct_meta: dict[int, dict[str, Any]],  # {traj_id: traj_dct_meta}
        w_size: int,
        n_sens_min: int,
        n_sens_max: int,
        n_epoch: int,
        samples_per_epoch: int,
        inp_buff: int,
        n_traj_train: int = 30,
        n_traj_burst: int = 4,
        windows_per_traj: int = 80,     # burst depth: how many windows per trajectory before switching
        seed: int = 0,
        cache_size: int = 100,
        train: bool = True,
        dynamic_sensor_sampling: bool = False,
        biased_sampling: bool = True,
        inter_samp_space: int = 1,
        i_sens_val: int = 2, # reference sensor index for validation given the minimum number of available sensors used. The actual index if n_sens_min + i_sens_val
        training_on_all_sens=False,
        exact_n_sensors: int | None = None,
        target_dt: float = 1.953125e-4,
        runs_dir: str = "data/runs3",
    ):
        self.dct_meta = dct_meta 
        self.w_size = int(w_size)
        self.n_sens_min = int(n_sens_min)
        self.n_sens_max = int(n_sens_max)
        self.samples_per_epoch = int(samples_per_epoch)
        self.inp_buff = inp_buff
        self.n_epoch = n_epoch
        self.sens_samp_prob = torch.zeros(n_sens_max +1 - n_sens_min)
        self.sens_samp_rates_init = torch.zeros(n_sens_max + 1 - n_sens_min)
        self.dynamic_sensor_sampling = dynamic_sensor_sampling
        self.i_sens_val = i_sens_val
        self.training_on_all_sens = training_on_all_sens
        self.exact_n_sensors = None if exact_n_sensors is None else int(exact_n_sensors)

        self.n_traj = int(n_traj_train)
        self.n_traj_burst = n_traj_burst
        self.windows_per_traj = int(windows_per_traj)

        self.seed = int(seed)
        self.seed_work = None
        self.fold_seed = mp.Value('i', 0)
        self.epoch_i = mp.Value('i', 0)

        self.old_fold_seed = self.fold_seed.value

        self.rng = None
        self.cache_size = int(cache_size)
        self.train = train
        self.target_dt = float(target_dt)
        self.runs_dir = runs_dir
        self.rng_train_traj = np.random.default_rng(seed)
        self.all_ids = np.array(sorted(self.dct_meta.keys()), dtype=np.int64)
        self.total_used_ids_train = np.array([], dtype=np.int64)
        self.sampling_space = np.arange(1, inp_buff + 1, inter_samp_space)

        if biased_sampling:
            self.prob_samp_space = np.linspace(1, 0, self.sampling_space.shape[0])
        else:
            self.prob_samp_space = np.array([1]*self.sampling_space.shape[0])

        if self.n_traj > len(self.all_ids):
            raise ValueError(f"n_traj ({self.n_traj}) to choose from for training > total trajectories ({len(self.all_ids)})")
        
        # Init of train and val traj pools, if cross val, init for the first fold
        traj_pool = self.rng_train_traj.choice(self.all_ids, n_traj_train, replace=False)
        if train:
            self.traj_pool = traj_pool
        else:
            self.traj_pool_train = traj_pool
            self.traj_pool = np.setdiff1d(self.all_ids, traj_pool, assume_unique=True)


    def __iter__(self):

        # only used for cross validation hyperparameter tuning to sample different trajectories for each fold.
        with self.fold_seed.get_lock():
            fold_seed = self.fold_seed.value

        if self.dynamic_sensor_sampling:
            # Adaptative sampling as training progresses. Beginning with training with the largest number of sensors. Increasing sampling rate of lower number of sensors as training progresses

            # used to keep track of epoch index to adapt sensor sampling rates.
            with self.epoch_i.get_lock():
                epoch_i = self.epoch_i.value
            if self.train:
                if epoch_i == 0:
                    self.init_sens_samp_prob()
                sens_samp_rates = (self.sens_samp_prob*self.sens_samp_rates_init**epoch_i) / (self.sens_samp_prob*self.sens_samp_rates_init**epoch_i).sum()
                sens_samp_rates = self.sens_samp_rates_init

        else:
            n_sens = self.n_sens_max - self.n_sens_min + 1
            if self.train and self.training_on_all_sens:
                # all number of sensors have same probability of being drawn
                sens_samp_rates = np.array([1/n_sens]*n_sens)
            elif self.train:
                sens_samp_rates = np.zeros(n_sens)
                sens_samp_rates[:self.i_sens_val+1] = 1.0/(self.i_sens_val + 1)
            else:
                sens_samp_rates = np.zeros(n_sens)
                sens_samp_rates[:self.i_sens_val+1] = 1.0/(self.i_sens_val + 1) # validate on specific variable number of sensors


        # Worker-aware: split samples_per_epoch across workers
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
            n_target = self.samples_per_epoch
        else:
            worker_id, num_workers = info.id, info.num_workers
            per_worker = (self.samples_per_epoch + num_workers - 1) // num_workers
            start = worker_id * per_worker
            end = min(start + per_worker, self.samples_per_epoch)
            n_target = max(0, end - start)

        # Used for cross validation, which is not the default setting
        if self.old_fold_seed != fold_seed:
            if self.train:
                self.total_used_ids_train = np.concatenate([self.total_used_ids_train, self.traj_pool])
            else:
                self.total_used_ids_train = np.concatenate([self.total_used_ids_train, self.traj_pool_train])

            eligible_ids = np.setdiff1d(self.all_ids, self.total_used_ids_train, assume_unique=True)
            new_train_traj_pool = self.rng_train_traj.choice(eligible_ids, self.n_traj, replace=False)
            if self.train:
                self.traj_pool = new_train_traj_pool
            else:
                self.traj_pool_train = new_train_traj_pool
                self.traj_pool = np.setdiff1d(self.all_ids, new_train_traj_pool, assume_unique=True)

            self.old_fold_seed = fold_seed

        
        # setting up new random generator for each epoch, not applied to validation
        if self.rng is None:
            self.seed_work = self.seed + 10214 * worker_id + fold_seed * 100000
            rng = np.random.default_rng(self.seed_work)
            if self.train: # train case
                self.rng = rng
        else: # never executed for validation
            rng = self.rng

        # keep a defined number of traj in cache to load them faster if they are randomly picked again
        cache = OrderedDict()

        def get_traj(tid: int):
            if tid in cache:
                cache.move_to_end(tid)
                return cache[tid]
            tens = torch.load(f'{self.runs_dir}/sim_{tid}.pt', mmap=True)
            # Optional sanity checks
            if tens.ndim != 2:
                raise ValueError(f"traj {tid} expected 2D (T,C), got {tens.shape}")
            if tens.shape[1] != self.n_sens_max:
                raise ValueError(f"traj {tid} has C={tens.shape[1]}, expected {self.n_sens_max}")
            cache[tid] = tens
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
            return tens

        def eligible_traj():
            eligible = []
            for tid in self.traj_pool:
                o = self.dct_meta[tid]['onset_ts']
                if o > self.inp_buff + self.w_size: 
                    eligible.append(int(tid))
            return eligible

        eligible = []

        trial = 1
        while not eligible:
            if trial > 50:
                break
            eligible = eligible_traj()
            if not eligible or len(eligible) < self.n_traj_burst:
                if not self.train:
                    raise ValueError('No/not enough valid trajectories to use for validation')
                # to modify because posible trajectory contamination between train/val
                self.traj_pool = self.rng_train_traj.choice(self.all_ids, self.n_traj, replace=False)
            trial += 1

        if not eligible:
            raise RuntimeError("Can't find any valid trajectories in the whole dataset")

        eligible = np.array(eligible, dtype=np.int64)

        rng.shuffle(eligible)

        eligible = eligible[worker_id::num_workers]

        if eligible.size == 0:
            print(f'Worker {worker_id} has no eligible trajectories to choose from!')
            return

        yielded = 0

        
        while yielded < n_target:
            
            # choose a few trajectories for locality
            size = min(self.n_traj_burst, eligible.size)
            pick = rng.choice(eligible.size, size=size, replace=False)
            trajs = eligible[pick]

            # burst: take several windows per trajectory before moving on
            for tid in trajs:
                traj = get_traj(tid)
                onset = self.dct_meta[tid]['onset_ts']
               
                if yielded >= n_target:
                    break
                if traj.shape[0] <= onset:
                    continue
                
                if onset <= self.w_size:
                    continue
                
                for _ in range(self.windows_per_traj):
                    if yielded >= n_target:
                        break

                    # k_end is the index of the last time step of the window to be retrieved
                    dist_to_onset = rng.choice(self.sampling_space, 1, p=self.prob_samp_space/self.prob_samp_space.sum())[0]
                    k_end = onset - dist_to_onset

                    # normalizing on time steps up to and including the last time step of the window.
                    traj_mean = traj[onset - self.inp_buff - self.w_size: k_end].mean(dim=0)
                    traj_std = traj[onset - self.inp_buff - self.w_size: k_end].std(dim=0)

                    label = math.log(float(onset - k_end))

                    x = traj[k_end - self.w_size: k_end, :].contiguous()        # (w_size, C) from disk
                    x = (x - traj_mean) / traj_std 

                    # rotation invariance
                    rotation_lag = int(rng.integers(0, self.n_sens_max))
                    x = torch.roll(x, rotation_lag, dims=1)
                    if self.exact_n_sensors is not None:
                        n_sens = self.exact_n_sensors
                    elif self.dynamic_sensor_sampling:
                        n_sens = rng.choice(self.n_sens_max+1 - self.n_sens_min, 1, p=sens_samp_rates) + self.n_sens_min
                    else:
                        n_sens = rng.integers(self.n_sens_min, self.n_sens_max + 1)
                    miss_ind = rng.choice(self.n_sens_max, self.n_sens_max - n_sens, replace=False)
                    miss_ind = torch.as_tensor(miss_ind, dtype=torch.long)
                    msk_ind = torch.ones(self.n_sens_max, dtype=torch.bool)
                    msk_ind[miss_ind] = False
                    x.mul_(msk_ind.to(dtype=x.dtype))

                    yielded += 1
                    yield x, msk_ind, torch.tensor(label, dtype=torch.float32)  # one example
        
    # only used for cross validation 
    def set_fold_seed(self, e: int):
        with self.fold_seed.get_lock(): 
            self.fold_seed.value = e

    # if we want dynamical sampling in terms of number of sensors available to the model. 
    # From experiments, this does not improve model performance
    def init_sens_samp_prob(self):
        amp_num_sens = self.n_sens_max - self.n_sens_min
        sens_samp_prob = torch.tensor([1/(2**i) for i in range(amp_num_sens + 1, 0, -1)])
        sens_samp_prob /= sens_samp_prob.sum()
        rate_min_sens = math.exp(math.log(0.2/sens_samp_prob[0])/self.n_epoch) - 1
        diff = (rate_min_sens * 3) / amp_num_sens
        sens_samp_rates = torch.tensor([rate_min_sens - diff*i for i in range(0, amp_num_sens+1)]) + 1

        self.sens_samp_prob = sens_samp_prob
        self.sens_samp_rates_init = sens_samp_rates
