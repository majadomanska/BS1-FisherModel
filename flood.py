import numpy as np
from strategies import EnvironmentDynamics


class ShockEnvironment(EnvironmentDynamics):
    """
    Environment with:
    - linear drift (like baseline)
    - small Gaussian noise every generation
    - occasional large random shock every T_shock generations
    """

    def __init__(self, alpha_init: np.ndarray,
                 c: np.ndarray,
                 delta: float,
                 T_shock: int,
                 sigma_shock: float):

        self.alpha = alpha_init.copy()
        self.c = np.array(c)
        self.delta = delta

        self.T_shock = T_shock
        self.sigma_shock = sigma_shock

        self.t = 0

    # ------------------------------------------------------------
    # 1. update environment state
    # ------------------------------------------------------------
    def update(self) -> None:

        #baseline drift
        noise = np.random.normal(
            loc=0.0,
            scale=self.delta,
            size=self.alpha.shape
        )

        #alpa = fluctuations
        self.alpha = self.alpha + self.c + noise

        #flood - adding a displacement
        if self.T_shock > 0 and self.t % self.T_shock == 0 and self.t > 0:
            
            #second arg is std (numpy), so the variance is smaller (=sigma**4)
            displacement = np.random.normal(0,self.sigma_shock**2, size=self.alpha.shape)

            self.alpha += displacement

        self.t += 1

    # ------------------------------------------------------------
    # 2. return optimum
    # ------------------------------------------------------------
    def get_optimal_phenotype(self) -> np.ndarray:
        return self.alpha