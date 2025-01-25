from modules.distributions.distribution_base_class import Distribution

class ScaledNormal(Distribution):
  def __init__(self, dim: int):
    super(ScaledNormal, self).__init__(dim)
    self.dim = dim

  def sample(self,):
    pass

  def log_likelihood(self,):
    pass

  def kl_divergence(self,):
    pass