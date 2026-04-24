import os
import json


def read_config(file_name):
    """Load a JSON configuration file."""
    with open(file_name, "r") as f:
        return json.load(f)


class initParams(object):
    """
    Set parameters and ensure working conditions for starting simulation.

    Parameters from Chaudhuri, He and Wang 2017:
    N = 440, tau = 195ms, gamma = 0.1 Hz/pA, mu_conn = 49.881 pA/Hz,
    sigma_conn = 4.988 pA/Hz, p = 0.2, and alpha = 1/44.
    """

    def __init__(
        self,
        N=400,
        prb=0.20,
        dt=0.1,
        dt_record=1,
        stabilize_time=150,
        total_time=350,
        noise_factor_add=0.0,
    ):
        assert stabilize_time < total_time, "total time can not be less than stabilize time"

        # Network parameters
        self.N = N
        self.dt = dt
        self.fs = (1 / dt_record) * 1000
        self.stabilize_time = stabilize_time

        # Time scale parameters
        self.time = total_time
        self.T = self.fs * self.time
        self.t_input_off = self.T

        # Model parameters
        self.alpha: 1 / 44
        self.gamma = 0.1
        self.prb = prb
        self.mu = 49.881
        self.sigma = 4.988
        self.tau = 195.0
        self.cap = 1000

        # Noise parameters
        self.noise_factor_add = noise_factor_add
        self.noise_factor_mult = 1.0
        self.low_pass_noise = False
        self.high_pass_noise = False

        # Block design parameters enabling change of setting within block
        self.block_size = 0
        self.gamma1 = self.gamma
        self.gamma2 = self.gamma
        self.noise_factor_add1 = noise_factor_add
        self.noise_factor_add2 = noise_factor_add

        # Prepare path and results and figures
        if not os.path.exists("figures"):
            os.makedirs("figures")
        if not os.path.exists("results"):
            os.makedirs("results")

        self.main_tag = "1"
        self.secondary_tag = "01"
        self.experiment_folder = f"Rate_model_{self.main_tag}"
        self.path = f"figures/{self.experiment_folder}/{self.secondary_tag}/"

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # Analysis params
        self.start_sample = self.fs * self.stabilize_time
        self.dt_record = dt_record
        self.change_factor = 0.01

    def add_to_noise_factor(self, delta) -> None:
        self.noise_factor += delta

    def __str__(self):
        return f"N={self.N}, prb={self.prb}"
