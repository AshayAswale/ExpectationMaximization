import csv
import sys
import numpy as np
import random
from copy import deepcopy
from numpy import genfromtxt
from scipy import linalg
from em_final import EM
# from scipy.spatial import distance
# from matplotlib import pyplot
import math
import time

def generate_random_pts(noc):
    dimension=2
    m=[]
    c=[]
    for i in range(noc):
        if(i>0):
            mean=mean+np.random.randint(1,5)
            pts=np.vstack((pts,np.random.multivariate_normal(mean,covariance,(25-noc)*20)))
        else:
            mean=np.random.rand(dimension)*np.random.randint(100,size=dimension)
            rm=mean
            covariance=np.diag(np.random.rand(dimension)*np.random.randint(10,size=dimension))
            pts=np.random.multivariate_normal(mean,covariance,(25-noc)*20)
        m.append(mean)
        c.append(covariance)
    print("Original Data Means:\n",np.asarray(m),"\n")
    return pts


def main(argv):
    # STORING GIVEN POINTS
    points = genfromtxt(argv[0], delimiter=" ")
    
    # points=generate_random_pts(10)
    # print(len(points))

    # INITIALIZING PARAMETERS
    best_means=[]
    best_cov=[]
    best_ll=-(math.inf)
    best_bic=math.inf
    n=int(argv[1])
    prog_start_time=time.time()

    # CHECKING WHICH MODES TO FOLLOW:
        # n=0 for finding best-fit number of clusters
        # n!=0 for finding best-fit clusters

    if n==0:
        threshold=1 # Thershold increased so as to accomodate more time for each cluster
        runtime=np.arange(2,21) # Runtime proportionally distributed according to the number of clusters.
        runtime=10*runtime/runtime.sum() # Total runtime accounts to 10 seconds.
        i=2 
        r=21
    else:
        threshold=0.01 # Thershold reduced so as to reach more better cluster estimates
        rt=10 # Runtime of 10 seconds
        i=n
        r=n+1
    while i <r:
        # print(i)
        # INITIALIZING LOCAL CLUSTER PARAMETERS
        c_ll=-math.inf 
        c_m=[]
        c_c=[]
        start_time=time.time() # Start-time for each cluster iteration
        if n==0: rt=runtime[i-2] 

        # LOOPING FOR BEST CUSTER ESTIMATES
        while time.time()-start_time <rt:
            pll=-(math.inf)
            ll=0
            test=EM(points,i) 
            while abs(ll-pll)>threshold:
                pll=ll
                e,ll=test.E_step() # Calculating expectations and log-likelihood
                # print(ll)
                if ll==0: # Avoiding errors
                    ll=c_ll
                    break
                test.M_step(e) # Maximizing the cluster estimates
                # print(ll)
            if ll>c_ll: # Prioritizing best cluster estimates
                c_ll=ll
                c_m=deepcopy(test.means)
                c_c=deepcopy(test.cov)
        # CALCULATING BIC INDEX
        bic=test.BIC(c_ll) 
        # print(bic)

        # PRIORITIZING CLUSTER WITH BEST BIC INDEX
        if bic<best_bic:
            best_bic=bic
            best_ll=c_ll
            best_means=c_m
            best_cov=c_c
            best_cluster=i
        i+=1

    #################################################################
    ############### P R I N T I N G    R E S U L T S ################
    #################################################################
    print("\nBest Likelihood: ",best_ll)
    print("\nBest BIC: ",best_bic)
    print("\nBest Cluster: ",best_cluster)
    print("\nBest Means:\n",best_means)
    print("\nBest Covariance:\n",best_cov)
    print("\nEnd-Time: %0.2f"%(time.time()-prog_start_time))



if __name__ == "__main__":
    main(sys.argv[1:])