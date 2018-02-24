import numpy as np
from scipy.sparse import csc_matrix
import json

def pageRank(G, s = .85, maxerr = .0001):
    """
    Computes the pagerank for each of the n states

    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.

    s: probability of following a transition. 1-s probability of teleporting
       to another state.

    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G,dtype=np.float)

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)

    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0,n):
            # inlinks of state i
            Ai = np.array(A[i].todense())[0]
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = np.array( Ai*s + Ei*(1-s) ).dot(ro)

    # return normalized pagerank
    return r/float(sum(r))

def pagerankMain(outputFile):
    contextNetworkData = json.load(open(outputFile.replace('CFN-PageRank', 'CFN-Vertex-Degree'), 'r'))

    for contextNetworks, filepath in contextNetworkData:
        for hypernym, contextNetwork in contextNetworks:
            contextNetwork = contextNetwork['key']
            mappingTable = {}
            invertedIndex = {}
            matrix = np.zeros((len(contextNetwork),len(contextNetwork)))

            for index, key in enumerate(contextNetwork):
                mappingTable[key] = index
                invertedIndex[index] = key

            for key, value in contextNetwork.items():
                matrix[[mappingTable[i] for i in value], mappingTable[key]] = 1
                matrix[mappingTable[key] , [mappingTable[i] for i in value]] = 1

            for key, value in contextNetwork.items():
                m, indegree = min([(i, len(contextNetwork[i])) for i in value], key=lambda x:x[1], default=(0, 0))
                if m == 0:
                    continue
                matrix[mappingTable[m]][mappingTable[key]] = 0

            rsums = matrix.sum(1)
            matrix /= rsums[:, np.newaxis]
            for index, _ in enumerate(matrix):
                if np.isnan(matrix[index]).all():
                    matrix[index] = 0

            for index, score in enumerate(pageRank(matrix)):
                contextNetwork[invertedIndex[index]] = score
    json.dump(contextNetworkData, open(outputFile, 'w'))

if __name__=='__main__':
    # Example extracted from 'Introduction to Information Retrieval'
    G = np.array([[0.5, 0.5, 0],
                  [0.5, 0, 0],
                  [0, 0.5, 1]])
    print(pageRank(G,s=.8))
    pagerankMain()