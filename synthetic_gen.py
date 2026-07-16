import numpy as np
import torch
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os
import scipy.signal as sig
import argparse
from pathlib import Path


PATH= Path.cwd()

def _rand_around(rng, mean, range_noise, n):
    range_min, range_plus = 1 - range_noise, 1 + range_noise
    low  = range_min * mean
    high = range_plus * mean
    if low > high:
        low, high = high, low
    return rng.uniform(low, high, n)


def randomize_init_burn_params(
        mean_om,
        mean_mu,
        mean_beta_r, mean_beta_i,
        mean_k_r, mean_k_i,
        n_burn,
        rng,
        range_noise = 0.02
):  
    
    range_min, range_plus = 1 - range_noise, 1 + range_noise
    dct = {}

    dct['om'] = _rand_around(rng, mean_om, range_noise, n_burn)

    dct['mu'] =  _rand_around(rng, mean_mu, range_noise, n_burn)

    beta_r = rng.uniform(range_min*mean_beta_r, range_plus*mean_beta_r, n_burn)
    beta_i = rng.uniform(range_min*mean_beta_i, range_plus*mean_beta_i, n_burn)
    beta = beta_r + 1j*beta_i
    dct['beta'] = beta

    k_r = rng.uniform(range_min*mean_k_r, range_plus*mean_k_r, n_burn)
    k_i = rng.uniform(range_min*mean_k_i, range_plus*mean_k_i, n_burn)
    k = k_r + 1j*k_i
    dct['k'] = k

    return dct


class full_combustor:
    def __init__(
            self,
            i_sim,
            n_burn: int,
            T,
            dt,
            seed,
            n_sensors,
            out_dir,
            random_burner_range = 0.07,
            noise_level= 0.4
    ):
        low, high = 1 - random_burner_range, 1 + random_burner_range
        # global simulation parameters
        self.out_dir = out_dir
        self.i_sim = i_sim
        self.n_burn = n_burn
        self.T = T
        self.t = 0
        self.dt = dt
        self.n_step = int(np.round(T / dt))
        self.t_n = np.arange(self.n_step) * self.dt


        self.rng = np.random.default_rng(seed=seed)
        
        self.mu_stable= -3.5*(1 + self.rng.uniform(-noise_level, noise_level))
        self.mu_sat_0 = 5*(1 + self.rng.uniform(-noise_level, noise_level))
        self.mu_sat = self.rng.uniform(self.mu_sat_0*low, self.mu_sat_0*high, n_burn)
        self.om_0 = 4712.39*(1 + self.rng.uniform(-noise_level, noise_level))
        self.mean_k_r = 1.5
        self.mean_k_i = 18

        # burner params 
        dct_params = randomize_init_burn_params(
            mean_om=self.om_0,  
            mean_mu=self.mu_stable,
            mean_beta_r=self.mu_sat, mean_beta_i=5,
            mean_k_r=self.mean_k_r, mean_k_i=self.mean_k_i,
            n_burn=n_burn,
            rng=self.rng,
            range_noise=random_burner_range
        )

        self.om = dct_params['om']
        self.mu_0 = dct_params['mu']
        self.mu = self.mu_0.copy()
        self.beta = dct_params['beta']
        self.k = dct_params['k']

        # ramp params init 
        self.X_0 : float = self.rng.uniform(-noise_level, noise_level)
        self.X_c_0 = 0.6*(1 + self.rng.uniform(-noise_level, noise_level))
        self.X_max_0 = 1*(1 + self.rng.uniform(-noise_level, noise_level))
        self.m_min_0 = 0.02*(1 + self.rng.uniform(-noise_level, noise_level))
        self.sigma_min_0 = 0.1*(1 + self.rng.uniform(-noise_level, noise_level))
        self.m_max_0 = 0.3*(1 + self.rng.uniform(-noise_level, noise_level))
        self.sigma_max_0 = 0.25*(1 + self.rng.uniform(-noise_level, noise_level))
        self.D_X = 0.35*(1 + self.rng.uniform(-noise_level, noise_level))

        # ramp drift equation
        self.E_m_beg = 0.0001*(1 + self.rng.uniform(-noise_level, noise_level))
        self.E_m_mid = 0.001*(1 + self.rng.uniform(-noise_level, noise_level))
        self.E_m = 1e-5#(self.E_m_beg + self.E_m_mid)/2 
        self.om_m = 4*(self.E_m_mid - self.E_m_beg)/1

        # ramp noise equation
        self.E_s_beg = 0.0001*(1 + self.rng.uniform(-noise_level, noise_level)) 
        self.E_s_mid = 0.001*(1 + self.rng.uniform(-noise_level, noise_level))
        self.E_s = 1e-5#(self.E_s_beg + self.E_s_mid)/2 
        self.om_s = 4*(self.E_s_mid - self.E_s_beg)/1 # Used to be 8

        # burner specific ramp arrays
        self.X = _rand_around(self.rng, self.X_0, random_burner_range, n_burn)
        self.X_c : np.ndarray = self.rng.uniform(self.X_c_0*low, self.X_c_0*high, n_burn)
        self.X_max = self.rng.uniform(self.X_max_0*low, self.X_max_0*high, n_burn)
        self.m_min = _rand_around(self.rng, self.m_min_0, random_burner_range, n_burn)
        self.m_max = self.rng.uniform(self.m_max_0*low, self.m_max_0*high, n_burn)
        self.sigma_min = self.rng.uniform(self.sigma_min_0*low, self.sigma_min_0*high, n_burn)
        self.sigma_max = self.rng.uniform(self.sigma_max_0*low, self.sigma_max_0*high, n_burn)
        self.E_m = self.rng.uniform(self.E_m*low, self.E_m*high, n_burn)
        self.om_m = self.rng.uniform(self.om_m*low, self.om_m*high, n_burn)
        self.E_s = self.rng.uniform(self.E_s*low, self.E_s*high, n_burn)
        self.om_s = self.rng.uniform(self.om_s*low, self.om_s*high, n_burn)

        # noise params, randomized by trajectory
        self.tau_noise = 0.0075*(1 + self.rng.uniform(-noise_level, noise_level)) #0.0055
        self.noise_intens = 1.15*(1 + self.rng.uniform(-noise_level, noise_level)) # noise intensity, used to be 1.15
        self.zeta = np.zeros(n_burn, dtype=np.complex128)
        self.rho = np.exp(-self.dt/(2*self.tau_noise))

        # amplitude time stepping objects init
        self.p_out = np.zeros((self.n_step, n_sensors))
        self.B = np.array([1e-5]*n_burn, dtype=np.complex128)
        self.B_arr = np.zeros((self.n_step, n_burn), dtype=np.complex128)
        self.p_sens : np.ndarray

        self.onset = False
        self.dct_out = {
            'metadata': {
                'T': T,
                'dt': dt,
                'n_burn': n_burn,
                'n_sensor': n_sensors,
                'mu_stable': self.mu_0,
                'mu_sat': self.mu_sat,
                'mu': np.mean(self.mu),
                'om': np.mean(self.om),
                'beta': np.mean(self.beta),
                'k': np.mean(self.k),
                'X_0': self.X_0,
                'X_c_0': self.X_c_0,
                'X_max_0':self.X_max_0,
                'm_min_0':self.m_min_0,
                'sigma_min_0':self.sigma_min_0,
                'm_max_0':self.m_max_0,
                'sigma_max_0':self.sigma_max_0,
                'D_X':self.D_X,
                'E_m_beg':self.E_m_beg,
                'E_m_mid':self.E_m_mid,
                'E_s_beg':self.E_s_beg,
                'E_s_mid':self.E_s_mid

            },
            'onset_ts': None
        }

        # threshold onset detection
        self.p : np.ndarray
        self.pers_thre = 25000
        self.onset_thre = 0.75
        self.p_stable : int
        self.p_unstabl: int
        self.onset_ts = None
        self.stride = 10
        self.last_step = self.n_step
        self.lag_after_onset = 110000

        # sensor data 
        self.n_sensor = n_sensors
        self.burn_loc = np.arange(n_burn)*2*np.pi/n_burn
        self.max_n_sensors = 12
        self.sensors_loc = np.round(np.linspace(0, self.max_n_sensors, n_sensors + 1)[:-1])*2*np.pi/self.max_n_sensors 
        self.f_modes_w_num = np.arange(-int(((n_burn - 1)/2)), int((n_burn-1)/2) + 1) if n_burn % 2 == 1 else np.arange(-int(n_burn/2 - 1), int(n_burn/2) + 1)

        # sensor noise
        self.tau_noise_sens =  0.001*(1 + self.rng.uniform(-noise_level, noise_level)) 
        self.noise_intens_sens = 0.025*(1 + self.rng.uniform(-noise_level, noise_level))
        self.zeta_sens = np.zeros(n_sensors)

    
    def step_noise(self):

        w_real = self.rng.normal(0, 1, self.n_burn)
        w_imag = self.rng.normal(0, 1, self.n_burn)
        self.zeta = self.rho*self.zeta + np.sqrt(self.noise_intens*self.tau_noise*(1 - self.rho**2)/2)*(w_real + 1j*w_imag)


    def amplitude_function(self, B, step_noise=False):

        if step_noise:
            self.step_noise()
        amp = (np.clip(self.mu, self.mu_0, self.mu_sat) + 1j*(self.om - self.om_0))*B - self.beta*(np.abs(B)**2)*B + self.k*(np.roll(B, 1) + np.roll(B, -1) - 2*B) + self.zeta

        return amp


    def rk_4_integration(self):

        """
        RK4 with noise half stepping.
        """

        k_1 = self.amplitude_function(self.B)
        B2 = self.B + (self.dt/2)*k_1
        k_2 = self.amplitude_function(B2, step_noise=True)
        B3 = self.B + (self.dt/2)*k_2
        k_3 = self.amplitude_function(B3)
        B4 = self.B + self.dt*k_3
        k_4 = self.amplitude_function(B4, step_noise=True)

        self.B = self.B + (self.dt/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)


    def mu_upd(self):
        
        # update ramp drift and noise
        self.E = np.abs(self.B)**2
        self.m = self.m_min + (self.m_max - self.m_min)*0.5*(1 + np.tanh((self.E - self.E_m)/self.om_m))
        self.s = self.sigma_min + (self.sigma_max - self.sigma_min)*0.5*(1 + np.tanh((self.E - self.E_s)/self.om_s))

        # update ramp
        self.X = self.X + (self.m + self.D_X*(np.roll(self.X, -1) + np.roll(self.X, 1) - 2*self.X))*self.dt + self.s*np.sqrt(self.dt)*self.rng.normal(0, 1, self.n_burn)
        self.X = np.clip(self.X, self.X_0, self.X_max)
 

        # apply appropriate ramp mapping to mu for each burner given ramp value.
        burn_bef_ons =  self.X < self.X_c
        self.mu[burn_bef_ons] = self.mu_0[burn_bef_ons]*(1 - self.X[burn_bef_ons]/self.X_c[burn_bef_ons])
        burn_aft_ons = ~burn_bef_ons
        self.mu[burn_aft_ons] = self.mu_sat[burn_aft_ons]*(self.X[burn_aft_ons] - self.X_c[burn_aft_ons])/(self.X_max[burn_aft_ons] - self.X_c[burn_aft_ons])



    def get_onset(self, onset_thresh=0.03, pers_quantile=4000, pers_adjust=2000):

        """
        Automatic onset finder. 
        - Computes the 90% quantile of the absolute signal magnitude in the stable period and the unstable period. 
        - Computes the difference between those quantiles in order to have an dynamic onset flaggin threshold given the magnitude of the instability event. 
        - find the first time step of the signal for which 90% of time steps following it remains above threshold.
        - Refines the estimation by making sure than the detected time steps sits at the onset of a significant burst.

        Note: In the way it is currently tuned, this annotator is quite restrictive. It will discard most smooth paths to instabilities. 
        """

        if self.p_sens.shape[0] < pers_quantile:
            print(f'Onset happens too early to safely determine onset in sim {self.i_sim}!')
            return
        if type(self.p_sens) == np.ndarray:
            p_max_time = np.abs(self.p_sens).max(axis=1)
        else:
            p_max_time = np.abs(self.p_sens).max(axis=1).values.numpy()
        p_stable = np.quantile(np.abs(p_max_time[:pers_quantile]), 0.9)
        p_unstable =  np.quantile(np.abs(p_max_time[-pers_quantile:]), 0.9)
        state_diff = p_unstable - p_stable

        self.dct_out['state_diff'] = state_diff
        p_thresh = p_stable + onset_thresh*state_diff
        msk = (p_max_time > p_thresh).astype(int)
        window = np.ones(pers_quantile, dtype=int)
        sliding = np.convolve(msk, window, mode='valid')
        onset = np.argmax(sliding > 0.15 * pers_quantile).item()
        if onset == 0:
            return
        print(f"That's the found onset for sim {self.i_sim}: {onset}")
        begin = max(0, int(onset - 5e4))
        onset_settled = False
        trials = 0
        while not onset_settled:
            if trials > 5000:
                print(f"Can't find satisfactory onset conditions after 3000 trials for sim {self.i_sim}, discarded" )
                return
            if onset < 7000 or onset > self.p_sens.shape[0]:
                print("Won't have enough windows to train on, discarded or have too much of them")
                return
            std_stab = p_max_time[begin:begin+pers_adjust].std()

            std_before_onset = p_max_time[max(onset - pers_adjust, 0): onset].std()
            std_after_onset = p_max_time[onset: min(onset + pers_adjust, self.p_sens.shape[0])].std()
    
            if std_before_onset > std_stab*2.5:
                onset = onset - 5
                begin = max(0, int(onset - 5e4))
                trials += 1
            elif std_after_onset < std_stab*2.5:
                onset = onset + 5
                begin = max(0, int(onset - 5e4))
                trials += 1
            else:
                onset_settled = True
                print(f"That's the corrected onset for sim {self.i_sim}: {onset}")
   
        self.dct_out['onset_ts'] = int(onset)


    def save_sensor_data(self):
        
        """
        This function is used to transform each burner complex amplitude into real sensor signal. 
        - It uses discrete Fourier transform to find the complex Fourier coefficients from the complex amplitude B_arr.  
        - From this, we reconstruct the complex amplitude signal for each sensor locations stored in self.sensors_loc.
        - We construct real sensor noise at each step and for each sensor.
        - We are adding back the fast carrier frequency om_0 to get a signal with an appropriate dominant mode frequency for thermoacoustics
        - Then we try to find the onset time step automatically.
        - If an onset is found, we decimate the signal by a factor 10 and save it as well as the trajectory metadata.
        """
        
        # discrete Fourier transform
        E1 = np.exp(-1j * np.outer(self.burn_loc, self.f_modes_w_num))  # (n_burn, n_mode)
        a_m = (self.B_arr @ E1) / self.n_burn  # (n_step, n_mode)

        # inverse discrete Fourier transform 
        E2 = np.exp(1j * np.outer(self.sensors_loc, self.f_modes_w_num))  # (n_sensor, n_mode)
        B_m = a_m @ E2.T    # (n_step, n_sens)

        # Adding sensor noise 
        # time correlation
        rho = np.exp(-self.dt/(self.tau_noise_sens))

        # Gaussian white noise, defined for all steps and all sensor locations
        w = self.rng.normal(0, 1, (self.last_step, self.n_sensor))

        # sensor noise variance 
        var_sens_noise = np.sqrt(self.noise_intens_sens*self.tau_noise_sens*(1 - rho**2))

        # innovation part
        eps = var_sens_noise * w

        # decaying part
        powers = rho ** np.arange(self.last_step)
        zeta_sens = sig.fftconvolve(eps, powers[:, None], mode="full")[:self.last_step]
        T = self.dt * self.last_step
        dt_arr = np.linspace(0, T, self.last_step)

        # injecting back carrier frequency to slow time dynamics complex amplitude signal
        E3 = np.exp(1j * self.om_0*dt_arr)

        # taking the real part of complex amplitude and adding sensor noise to it
        self.p_sens = np.real(E3[:, None]*B_m) + zeta_sens

        # decimating by a factor 10 using filtering to handle aliasing
        self.p_sens = sig.decimate(self.p_sens, self.stride, ftype="fir", zero_phase=True, axis=0)
        print('here')
        # trying to automatically find onset 
        self.get_onset()

        # Automatic annotator could not find a satisfying onset
        if not self.dct_out['onset_ts']:
            return
        
        p_sens = torch.from_numpy(self.p_sens)

        # saving trajectory
        out_path_tens = f'{self.out_dir}/runs8'
        os.makedirs(out_path_tens, exist_ok=True)
        torch.save(p_sens, f'{out_path_tens}/sim_{self.i_sim}.pt')

        # saving trajectory metadata dict
        out_path_dct = f'{self.out_dir}/dct_meta8'
        os.makedirs(out_path_dct, exist_ok=True)
        torch.save(self.dct_out, f"{out_path_dct}/sim_{self.i_sim}.pth")


    def main(self):
        print(f'Starting sim {self.i_sim}!!')
        
        persis_onset = np.zeros(self.pers_thre)

        count_after_onset = 0
        for i in tqdm(range(self.n_step), desc="Simulating"):

            self.rk_4_integration()
            self.B_arr[i] = self.B
            self.mu_upd()

            # detecing onset internally using the value of the ramp variable. 
            self.onset = (self.X > self.onset_thre).sum() >= 1

            persis_onset = np.roll(persis_onset, 1)

            if self.onset:
                persis_onset[0] = 1
                if np.mean(persis_onset) > 0.9 and self.onset_ts is None:
                    self.onset_ts = i - self.pers_thre
            else:
                persis_onset[0] = 0
            
            if self.onset_ts:
                count_after_onset += 1

            # let the system evolve for a few thousand steps after onset detected to develop instability event.
            if count_after_onset > self.lag_after_onset:
                self.last_step = i
                self.B_arr = self.B_arr[:i]
                break

            self.t += self.dt

        # if onset is detected internally, triggering saving function.
        if self.onset_ts:
            self.save_sensor_data()



def main_wrapper(params):
    """Wrapper for multiprocessing; unpacks dictionary into `main`."""
    sim = full_combustor(**params)
    sim.main()
    return None


def main(
        n_sim,
        T,
        dt,
        seed ,
        multiburn,
        n_sens,
        out_dir
):
    
    np.random.seed(seed)
    seeds = np.random.randint(0, 10000, n_sim)

    # If we ever want to generate trajectory with a different number of burners. We need at least 3 burners for the coupling term.
    if multiburn:
        n_burner_arr = np.random.randint(2, 30, n_sim)

    # predefined number of burners.
    else:
        n_burn = 12
        n_burner_arr = np.array([n_burn] * n_sim)
    

    l_inp = [
        {   'i_sim': i_sim + 0,
            'seed': seed_i,
            'T': T,
            'dt': dt,
            'n_burn': n_burn,
            'n_sensors': n_sens,
            'out_dir': out_dir
        } 
        for i_sim, (seed_i, n_burn) in enumerate(zip(seeds, n_burner_arr))
        ]

    runmp = partial(main_wrapper)

    num_workers = min(32, n_sim)
    with Pool(num_workers) as pool:
        pool.map(runmp, l_inp, chunksize=1)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SL ROM")
    p.add_argument("-n_sim", required=True, type=int, help="number of simulated trajectories.")     
    p.add_argument("-out_dir", type=str, required=True, help="output dir for traj and metadata dict")     
    p.add_argument("-T", default=10, type=float, help="total simulation time.")   
    p.add_argument("-dt", default=1.953125e-05, type=float, help="difference of time between two time steps. Default value matches with Indlekofer experimental dataset sampling rate")
    p.add_argument("-multiburn", default=False, type=bool, help="Generating trajectories with different number of burners.")
    p.add_argument("-n_sens", default=12, type=int, help="Number of sensors.")


    return p                                                              


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    out_dir = args.out_dir
    seed = 0

    inp = {
        'n_sim': args.n_sim,
        'T': args.T,
        'dt': args.dt,
        'seed': seed,
        'multiburn': args.multiburn,
        'n_sens': args.n_sens,
        'out_dir': out_dir
    }

    main(**inp)

    l_sim_dct_file = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
    print(f'number of saved files: {len(l_sim_dct_file)}')

