import numpy as np
from numpy.random import randint
from scipy.sparse import csr_matrix

# (d x N) -> K -> ?(d x K) -> (N x K)
import numpy as np
from numpy.random import randint
from scipy.sparse import csr_matrix

# (d x N) -> K -> ?(d x K) -> (N x K)
def HardCMeans(DataSet, ClusterCount, InitialCenters = None, Precision = np.float_):
   DataSetSize        = DataSet.shape[1]
   Dimension          = DataSet.shape[0]

   # Init:
   OldClusterCenterMx = None
   ClusterCenterMx    = None
   
   OldClusterAssignMx = None
   ClusterAssignMx    = None

   # We can provide initial centers:
   if InitialCenters is None:
      print 'Initial Centers Randomized'
      InitialCenterIndices = randint(0, DataSetSize, ClusterCount)
      ClusterCenterMx      = np.array(DataSet[:,InitialCenterIndices], copy=True)
   else:
      print 'Initial Centers Provided'
      ClusterCenterMx = np.array(InitialCenters, copy=True)    

   RowIndices     = np.arange(0, DataSetSize)
   DSizeRankOnes  = np.ones(DataSetSize)
   DistanceMatrix = np.empty(shape=(DataSetSize, ClusterCount), dtype=Precision)

   while (((OldClusterAssignMx is None) or ((OldClusterAssignMx != ClusterAssignMx).nnz != 0)) and 
      ((OldClusterCenterMx is None) or not np.array_equal(OldClusterCenterMx, ClusterCenterMx))):
      print "Iteration..."
      OldClusterAssignMx = ClusterAssignMx
      OldClusterCenterMx = ClusterCenterMx
        
      # Computing Distance Matrix:
      np.dot(DataSet.T, ClusterCenterMx, out=DistanceMatrix)
      DistanceMatrix *= -2
      DistanceMatrix += np.sum(ClusterCenterMx ** 2, axis=0, keepdims=True)
    
      # Computing Closest Center:
      BestAssignments = np.argmin(DistanceMatrix, axis=1)

      # Computing Assignment Matrix
      ClusterAssignMx = csr_matrix((DSizeRankOnes, (RowIndices, BestAssignments)), shape=(DataSetSize, ClusterCount))
    
      # Computing New Centers:
      ClusterCenterMx = ClusterAssignMx.T.dot(DataSet.T).T
      ClusterCenterMx /= ClusterAssignMx.sum(axis=0)
    
   return ClusterAssignMx.toarray(), ClusterCenterMx


