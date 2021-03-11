import numpy as np
import pandas as pd
import matplotlib

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

def get_features(file_path):
    df=pd.read_csv(file_path)[0:100]
    y=np.array(df["selling_price"],dtype=float).reshape(-1,1)
    df=df.drop(["Index","name","torque","owner","selling_price","seller_type"],axis=1)

    for i in range(len(df["fuel"])):

        s= df["fuel"][i]
        df["fuel"][i] = 1.0 if (s == "Petrol") else 0.0

    for i in range(len(df["transmission"])):

        s= df["transmission"][i]
        df["transmission"][i] = 1.0 if (s == "Manual") else 0.0 
            

    for i in range(len(df["engine"])):

        s= df["engine"][i]
        if type(s) == str:
            if s.endswith(" CC"):
                res = s[:-(len(" CC"))]
                df["engine"][i] = float(res)

    for i in range(len(df["mileage"])):

        s= df["mileage"][i]
        if type(s) == str:
            if s.endswith(" kmpl"):
                res = s[:-(len(" kmpl"))]
                df["mileage"][i] = float(res)
            elif s.endswith(" km/kg"):
                res = s[:-(len(" km/kg"))]
                df["mileage"][i] = float(res)
    # print(df["engine"])

    for x in df.head():
        # for i in range(len(df[x])):
        df[x] = (df[x]).astype(float)
        df[x]=np.nan_to_num(df[x])
        norm = np.linalg.norm(df[x])
        # print(df[x][0], norm)
        df[x] = df[x]/norm

    phi=np.array(df)

    return phi,y	


def get_features_basis(file_path):
    df=pd.read_csv(file_path)
    y=np.array(df["selling_price"])
    phi=np.array(df.drop(["selling_price"],axis=1))
    return phi[0],y


def compute_RMSE(phi, w , y):

    yhat= phi.dot(w) 
    return np.sqrt(np.sum((yhat-y)**2)/phi.shape[0])

# def generate_output(phi_test, w):
# 	# writes a file (output.csv) containing target variables in required format for Submission.
    
	
def closed_soln(phi, y):

    return np.linalg.pinv(phi).dot(y)
	
def gradient_descent(phi, y, phi_dev, y_dev) :
	# Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    n,d=np.shape(phi)
    w=np.zeros((d,1))
    alpha=15

    for _ in range(100000):
        w -= alpha*phi.T.dot((phi.dot(w)-y))/n

    return w

def sgd(phi, y, phi_dev, y_dev) :
	# Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    n,d=np.shape(phi)
    w=np.zeros((d,1))
    alpha=1

    for _ in range(10):    
        batchindx= np.random.choice(n,int(n/100))

        phibatch= np.array([phi[i] for i in batchindx])
        ybatch=np.array([y[i] for i in batchindx])

        for _ in range(1000000):
            w -= alpha*phibatch.T.dot(phibatch.dot(w)-ybatch)/(n/10)

    return w


def pnorm(phi, y, phi_dev, y_dev, p) :
	# Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    n,d=np.shape(phi)
    w=np.zeros((d,1))
    alpha=0.01

    print(p)


    if p % 2 ==1:
        for _ in range(100):
            w -= alpha*(phi.T.dot((phi.dot(w)-y))/n - 0.0001*p*(w**(p-1))) 

    else:

        for _ in range(100):
            print(w)
            print(p)
            w -= alpha*(phi.T.dot((phi.dot(w)-y))/n - 0.0001*p*np.array([x**(p-1) if x>=0 else -x**(p-1) for x in w]))





    return w	


def main():

        ######## Task 1 #########
        phase = "train"
        phi, y = get_features('df_train.csv')
        
        phase = "eval"
        phi_dev, y_dev = get_features('df_val.csv')
        # w1 = closed_soln(phi, y)
        # w2 = gradient_descent(phi, y, phi_dev, y_dev)
        # r1 = compute_RMSE(phi_dev, w1, y_dev)        # w1 = closed_soln(phi, y)
        # w2 = gradient_descent(phi, y, phi_dev, y_dev)
        # r1 = compute_RMSE(phi_dev, w1, y_dev)
        # r2 = compute_RMSE(phi_dev, w2, y_dev)
        # print('1a: ')
        # print(abs(r1-r2))
        # w3 = sgd(phi, y, phi_dev, y_dev)
        # r3 = compute_RMSE(phi_dev, w3, y_dev)
        # print('1c: ')
        # print(abs(r2-r3))
        # r2 = compute_RMSE(phi_dev, w2, y_dev)
        # print('1a: ')
        # print(abs(r1-r2))
        # w3 = sgd(phi, y, phi_dev, y_dev)
        # r3 = compute_RMSE(phi_dev, w3, y_dev)
        # print('1c: ')
        # print(abs(r2-r3))

#         ######## Task 2 #########
        w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
        w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
#         r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
#         r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
#         print('2: pnorm2')
#         print(r_p2)
#         print('2: pnorm4')
#         print(r_p4)

#         ######## Task 3 #########
#         phase = "train"
#         phi_basis, y = get_features_basis1('train.csv')
#         phase = "eval"
#         phi_dev, y_dev = get_features_basis1('dev.csv')
#         w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
#         rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
#         print('Task 3: basis')
#         print(rmse_basis)

main()
# print(get_features("df_train.csv"))
