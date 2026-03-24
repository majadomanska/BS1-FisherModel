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
    # 1. update environment
    # ------------------------------------------------------------
    def update(self) -> None:

        #baseline drift
        noise = np.random.normal(
            loc=0.0,
            scale=self.delta,
            size=self.alpha.shape
        )
        
        #optimial location: (x,y) = (0.0, 0.0)
        noise[-2:] = 0.0
        c = self.c.copy()
        c[-2:] = 0.0

        #alpa = fluctuations (applies to location as well)
        self.alpha = self.alpha + self.c + noise

        #flood - adding a displacement (but not to the location!)
        if self.T_shock > 0 and self.t % self.T_shock == 0 and self.t > 0:
            
            #second arg is std (numpy), so the variance is = sigma**2
            displacement = np.random.normal(0,
                                            self.sigma_shock,
                                            size=self.alpha.shape
                                              )
            #not displacing location
            displacement[-2:] = 0.0
            self.alpha += displacement
            

        self.alpha[-1] = 0.0
        self.alpha[-2] = 0.0
        
        self.t += 1

    # ------------------------------------------------------------
    # 2. return optimum
    # ------------------------------------------------------------
    def get_optimal_phenotype(self) -> np.ndarray:
        return self.alpha