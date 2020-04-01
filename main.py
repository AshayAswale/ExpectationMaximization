import numpy as np
from time import time
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from em import EM
import time

def main(argv):
    # Reading and storing data
    my_file=open(argv[0])
    file_lines=my_file.readlines()
    # No of Clusters
    noc=int(argv[1])
    coordinates=[]
    random_restarts=10
    for i in range(len(file_lines)):
        file_lines[i]=file_lines[i].replace("\n","")
        file_lines[i]=file_lines[i].split(" ")
        coordinates.append([])
        for j in range(len(file_lines[i])):
            coordinates[i].append(float(file_lines[i][j]));
    # No of Dimensions
    n=len(coordinates[0])
    # Solver
    start_time=time.time()
    z=0  
    while(z<random_restarts and time.time()-start_time<50):
        cluster=EM(n,coordinates,noc)
        for i in range(20):
            # e= Expected Value.shape=(len(data),noc)
            e=cluster.expectation(coordinates)
            # Mean= Covariance Matrix=[Mean cluster1, Mean cluster2, Mean cluster3]
            mean=np.array(cluster.mean(e,coordinates))
            # Matrix= Covariance Matrix=[cov cluster1, cov cluster2, cov cluster3]
            matrix=cluster.covariance(e,coordinates,mean)
            # Cont= Prob(cluster)
            cont=np.array(sum(e))/len(coordinates)
            cl=[]
            for t in range(noc):
                cl.append([])
            for q in range(len(coordinates)):
                for w in range(noc):
                    if e[q][w]==e[q].max():
                        cl[w].append(coordinates[q])
            print(cluster.log_likelihood(coordinates,e))
            # Update Values
            for j in range(noc):
                print(mean[j],"\n")
                cluster.clusters["cluster %s"%(j+1)]["mean"]=mean[j]
                cluster.clusters["cluster %s"%(j+1)]["cov"]=matrix[j]
                cluster.clusters["cluster %s"%(j+1)]["cont"]=cont[j]

            # Plot Data 
            fig = plt.figure()
            ax = plt.axes()
            ax.scatter((np.array(cl[0]).T)[0],(np.array(cl[0]).T)[1])
            ax.scatter((np.array(cl[1]).T)[0],(np.array(cl[1]).T)[1])
            ax.scatter((np.array(cl[2]).T)[0],(np.array(cl[2]).T)[1])
            # ax.scatter((np.array(cl[3]).T)[0],(np.array(cl[3]).T)[1])
            # ax.scatter((np.array(cl[4]).T)[0],(np.array(cl[4]).T)[1])
            # ax.scatter((np.array(coordinates).T)[0],(np.array(coordinates).T)[1])
            ax.scatter(mean.T[0],mean.T[1],c=[0,1,2])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")


            plt.show() 
        
            # plt.close(fig)
        z+=1
        print("\n")        

    # print(cluster.clusters)
    # print("\n")
    # print(np.array(e))
    # print(mean)
    # print(matrix,"\n")

    # print(cluster.clusters)
    # print(mean)




if __name__ == "__main__":
    main(sys.argv[1:])
