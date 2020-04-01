import numpy as np
import random
class EM:
    def __init__(self,n,coordinates,noc):
        self.coordinates=coordinates
        self.dimensions=n
        self.noc=noc
        cont=np.random.random(self.noc)
        cont=cont/cont.sum()
        self.clusters={}
        for i in range(noc):
            j=np.random.randint(len(coordinates))
            k=np.random.randint(len(coordinates))
            # print(coordinates[j])
            self.clusters["cluster %d"%(i+1)]={"mean":np.array(coordinates[j]),"cov":np.diag(np.array(coordinates[k])*np.array(coordinates[k])),"cont":cont[i]}
        # print(self.clusters)

    def probability(self,coordinates,cluster):
        cov_inv=np.linalg.inv(cluster["cov"])
        mean_dist=coordinates-cluster["mean"]
        exponential_term=np.exp((-0.5)*np.dot(mean_dist,np.dot(cov_inv,mean_dist)))
        # print("EXP:",exponential_term)
        root_term=np.sqrt((((2*np.pi)**self.dimensions))*abs(np.linalg.det(cluster["cov"])))
        return exponential_term/root_term

    
    def log_likelihood(self,coordinates,expectation):
        e=[]
        for i in range(len(coordinates)):
            p=[]
            for j in range(len(self.clusters)):
                p.append(self.probability(coordinates[i],self.clusters["cluster %s"%(j+1)])*self.clusters["cluster %s"%(j+1)]["cont"])
            e.append(p)
        return np.log(np.array(e)).sum()
    def expectation(self,coordinates):
        e=[]
        for i in range(len(coordinates)):
            p=[]
            for j in range(len(self.clusters)):
                p.append(self.probability(coordinates[i],self.clusters["cluster %s"%(j+1)])*self.clusters["cluster %s"%(j+1)]["cont"])
            p=np.array(p)/sum(p)
            e.append(p)
        return e

    def mean(self,expectation,coordinates):
        m=[]
        for i in range(len(self.clusters)):
            esum=sum(expectation)
            mean=0
            for j in range(len(coordinates)):
                mean+=expectation[j][i]*np.array(coordinates[j])
            m.append(mean/esum[i])
        return m

    def covariance(self,expectation,coordinates,mean):
        matrix=[]        
        for i in range(len(self.clusters)):
            esum=sum(expectation)
            cov=0
            for j in range(len(coordinates)):
                dist=(np.array(coordinates[j])-mean[i]).reshape(1,self.dimensions)
                cov+=expectation[j][i]*np.dot(dist.T,dist)
            matrix.append(cov/esum[i])
        return np.array(matrix)