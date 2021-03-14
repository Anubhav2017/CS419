import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""
compdic={}
def create_list_from_dic(x):


    phicomp=[]

    for item in x:
        comp=item.partition(' ')[0]
        phicomp.append(np.array(compdic[comp]))

    phicomp=np.array(phicomp)

    return phicomp
 
def create_dic_names(x):

    global compdic

    comps=[]

    for item in x:
        comp=item.partition(' ')[0]
        if comp not in comps:
            comps.append(comp)

    ncomp= len(comps)

    for i in range(ncomp):
        val= [0 for _ in range(ncomp)]
        val[i]=1

        compdic[comps[i]]=val

def get_features(file_path):
    df=pd.read_csv(file_path)
    
    df=df.drop(["Index","torque","transmission","fuel","owner","seller_type","transmission","mileage"],axis=1)

    # for i in range(len(df["fuel"])):

    #     s= df["fuel"][i]
    #     df["fuel"][i] = 1.0 if (s == "Petrol") else 0.0

    # for i in range(len(df["transmission"])):

    #     s= df["transmission"][i]
    #     df["transmission"][i] = 1.0 if (s == "Manual") else 0.0 
            

    for i in range(len(df["engine"])):

        s= df["engine"][i]
        if type(s) == str:
            if s.endswith(" CC"):
                res = s[:-(len(" CC"))]
                df["engine"][i] = float(res)

    # for i in range(len(df["mileage"])):

    #     s= df["mileage"][i]
    #     if type(s) == str:
    #         if s.endswith(" kmpl"):
    #             res = s[:-(len(" kmpl"))]
    #             df["mileage"][i] = float(res)
    #         elif s.endswith(" km/kg"):
    #             res = s[:-(len(" km/kg"))]
    #             df["mileage"][i] = float(res)
    # print(df["engine"])
    # print(df.head())

    for x in df.head():
        if x != "name":
            # for i in range(len(df[x])):
            df[x] = (df[x]).astype(float)
    
    df=df.dropna()
    phicomp= create_list_from_dic(df["name"])
    df=df.drop(["name"],axis=1)
    y=np.array(df["selling_price"],dtype=float).reshape(-1,1)
    df = df.drop(["selling_price"],axis=1)


    # create_list_names()


    for x in df.head():
        # df[x]=np.nan_to_num(df[x])
        norm = np.linalg.norm(df[x])
        df[x] = df[x]/norm
        # print(df[x])

    phi=np.array(df)

    phi= np.concatenate((phi,phicomp),axis=1)

    return phi,y	


def get_features_basis(file_path):
    df=pd.read_csv(file_path)
    
    df=df.drop(["Index","torque","transmission","fuel","owner","seller_type","transmission","seats","mileage","seats","year"],axis=1)

    # for i in range(len(df["fuel"])):

    #     s= df["fuel"][i]
    #     df["fuel"][i] = 1.0 if (s == "Petrol") else 0.0

    # for i in range(len(df["transmission"])):

    #     s= df["transmission"][i]
    #     df["transmission"][i] = 1.0 if (s == "Manual") else 0.0 
            

    for i in range(len(df["engine"])):

        s= df["engine"][i]
        if type(s) == str:
            if s.endswith(" CC"):
                res = s[:-(len(" CC"))]
                df["engine"][i] = float(res)

    # for i in range(len(df["mileage"])):

    #     s= df["mileage"][i]
    #     if type(s) == str:
    #         if s.endswith(" kmpl"):
    #             res = s[:-(len(" kmpl"))]
    #             df["mileage"][i] = float(res)
    #         elif s.endswith(" km/kg"):
    #             res = s[:-(len(" km/kg"))]
    #             df["mileage"][i] = float(res)
    # print(df["engine"])
    # print(df.head())

    for x in df.head():
        if x != "name":
            # for i in range(len(df[x])):
            df[x] = (df[x]).astype(float)
    
    df=df.dropna()
    phicomp= create_list_from_dic(df["name"])
    df=df.drop(["name"],axis=1)
    y=np.array(df["selling_price"],dtype=float).reshape(-1,1)
    df = df.drop(["selling_price"],axis=1)


    # create_list_names()


    for x in df.head():
        # df[x]=np.nan_to_num(df[x])
        norm = np.linalg.norm(df[x])
        df[x] = df[x]/norm
        # print(df[x])

    phi=np.array(df)

    phi= np.concatenate((phi,phicomp),axis=1)

    return phi,y	


def compute_RMSE(phi, w , y):

    yhat= phi.dot(w) 
    return np.sqrt(np.sum((yhat-y)**2)/phi.shape[0])

def d_abs(w,p):
    result=[]
    for x in w:
        mask1 = (x>= 0)*p*(x**(p-1))
        mask2 = (x< 0)*(-p)*(x**(p-1))
        result.append(mask1+mask2)

    return np.array(result)

	
def closed_soln(phi, y):

    return np.linalg.pinv(phi).dot(y)
	
def gradient_descent(phi, y, phi_dev, y_dev) :
	# Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    n,d=np.shape(phi)
    ndev= np.shape(phi_dev)[0]

    w=np.zeros((d,1))
    alpha=0.01

    losses=[]

    losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
    w -= alpha*phi.T.dot((phi.dot(w)-y))/n

    i=0

    while(losses[i] > 10):
        losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
        # print(losses[i])
        w -= alpha*phi.T.dot((phi.dot(w)-y))/n
        i+=1

    # plt.plot(losses)
    # plt.show()

    return w

def sgd(phi, y, phi_dev, y_dev) :
	# Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    n,d=np.shape(phi)
    # print(d)
    ndev= np.shape(phi_dev)[0]
    w=np.zeros((d,1))
    alpha=0.01

    losses = []
    j=0

    for _ in range(5):    

        nbatch= int(n/100)
        batchindx= np.random.choice(n,nbatch)
        # print(batchindx)
        # print(n)

        phibatch= phi[batchindx,:]
        ybatch=y[batchindx,:]

        losses=[]
        i=0
        losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
        w -= alpha*phibatch.T.dot(phibatch.dot(w)-ybatch)/nbatch

        for _ in range(int(15000/2**j)):
            losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
            # print(losses[i])
            w -= alpha*phibatch.T.dot((phibatch.dot(w)-ybatch))/nbatch
            i+=1

        j+=1

        # plt.plot(losses)
        # plt.show()


    return w


def pnorm(phi, y, phi_dev, y_dev, p) :
	# Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    n,d=np.shape(phi)
    w=np.zeros((d,1))
    alpha=0.05

    n,d=np.shape(phi)
    ndev= np.shape(phi_dev)[0]

    w=np.zeros((d,1))
    alpha=0.05

    losses=[]

    losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
    w -= alpha*phi.T.dot((phi.dot(w)-y))/n

    i=0

    if p==2:
        lmbda= 0.001
    else:
        lmbda = 10**(-17)

    while(losses[i] > 10000):
        losses.append(abs(np.sum((phi_dev.dot(w)-y_dev))/ndev))
        # print(losses[i])
        w -= alpha*(phi.T.dot((phi.dot(w)-y))/n + lmbda*p*w**(p-1))
        i+=1

    # plt.plot(losses)
    # plt.show()

    return w
    

def main():

        ######## Task 1 #########
        phase = "train"
        df1= pd.read_csv('df_train.csv')
        df2=pd.read_csv('df_val.csv')
        create_dic_names(np.concatenate((df1["name"],df2["name"]),axis=0))


        phi, y = get_features('df_train.csv')
        # print(phi)
        
        # phase = "eval"
        phi_dev, y_dev = get_features('df_val.csv')

        

        



        w1 = closed_soln(phi, y)
        w2 = gradient_descent(phi, y, phi_dev, y_dev)
        r1 = compute_RMSE(phi_dev, w1, y_dev)       
        r2 = compute_RMSE(phi_dev, w2, y_dev)
        # print('1a: ')
        print("r2:",abs(r2))
        print(abs(r1-r2))
        w3 = sgd(phi, y, phi_dev, y_dev)
        r3 = compute_RMSE(phi_dev, w3, y_dev)
        print('1c: ')
        print("r3:",abs(r3))
        print(abs(r3-r2))
  

#         ######## Task 2 #########
        w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
        w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
        r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
        r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
        print('2: pnorm2')
        print(r_p2)
        print('2: pnorm4')
        print(r_p4)

        ######## Task 3 #########
        phase = "train"
        phi_basis, y = get_features_basis('df_train.csv')
        phase = "eval"
        phi_dev, y_dev = get_features_basis('df_val.csv')
        w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
        rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
        print('Task 3: basis')
        print(rmse_basis)

main()
# print(get_features("df_train.csv"))
