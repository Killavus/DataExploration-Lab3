import numpy as np
from numpy.random import multinomial, normal

# TrivialCovariances = All Covariances are Eye Matrices, it simplifies the process a lot
# (1 x K) -> (K x d) -> (K x d x d) -> N -> Bool -> (d x N, 1 x N)
def SamplesFromGaussianMixture(Probs, Means, CovarianceMatrices, SampleCount, TrivialCovariances=False, Precision=np.float_, ChoicesPrecision=np.int_):
  MixtureCount = Probs.shape[0]  
  Dimension    = Means.shape[1]
  CholeskyMatrices = CovarianceMatrices
  if not TrivialCovariances:
      CholeskyMatrices = np.linalg.cholesky(CovarianceMatrices) # K x d x d
    
  # Means - K x d
  # CholeskyMatrices - # K x d x d
  
  ResultSet = np.empty(shape=(Dimension, SampleCount), dtype=Precision) # d x N
  Choices = np.zeros(SampleCount, dtype=ChoicesPrecision)

  MixturesToSample = multinomial(SampleCount, Probs)
  GeneratedSamples = 0
  for MixtureInd in range(MixtureCount):
     Count = MixturesToSample[MixtureInd]
     ZMatrix = normal(size=(Dimension, Count))
     if not TrivialCovariances:
       ZMatrix = np.dot(CholeskyMatrices[MixtureInd], ZMatrix)
     ResultSet[:, GeneratedSamples:(GeneratedSamples + Count)] = ZMatrix + Means[MixtureInd].reshape(Means[MixtureInd].shape[0], 1)
     Choices[GeneratedSamples:(GeneratedSamples + Count)] = MixtureInd
     GeneratedSamples += Count
 
  return ResultSet, Choices

