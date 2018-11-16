import numpy
import pandas as df
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *
import matplotlib.pyplot as plt

### Problem 1
stuff = df.read_csv('data.txt')
def nllike1(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
def nllike2(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*(obs.x)**2
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
def coolFunc(data):
    initialGuess=numpy.array([1,1,1])
    fit1=minimize(nllike1,initialGuess,method="Nelder-Mead",options={'disp': False},args=data)
    initialGuess=numpy.array([1,1,1,1])
    fit2=minimize(nllike2,initialGuess,method="Nelder-Mead",options={'disp': False},args=data)
    teststat=2*(fit2.fun-fit1.fun)
    mrah=len(fit2.x)-len(fit1.x)
    
    if 1-chi2.cdf(teststat,mrah) >0.05:
        return "Linear is a more suitable model"
    else:
        return "Quadratic is a more suitable model"
coolFunc(stuff)

### Problem 2
import scipy
import scipy.integrate as spint
def paperSim(y,t0,r1,r2,alpha11,alpha12,alpha21,alpha22):
    N1 = y[0]
    N2 = y[1]
    dN1dt=r1*(1-alpha11*N1-alpha12*N2)*N1
    dN2dt=r2*(1-N2*alpha22-alpha21*N1)*N2
    return [dN1dt, dN2dt]


###r1,r2,K1,K2,alpha1,alpha2
N0=[0.1,0.1]
times=range(0,100)


### Case 1 alpha11>alpha12 and alpha22>alpha21
N0=[0.1,0.11]
###r1,r2,alpha11,alpha12,alpha21,alpha22
params = (0.1,0.1,2,1,1,2)
modelSim=spint.odeint(func=paperSim,y0=N0,t=times,args=params)
simDF=df.DataFrame({"t":times,"normal":modelSim[:,0],"tumor":modelSim[:,1]})
case1= ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
case1.draw()

### Case 2 alpha12>alpha11 and alpha22>alpha21
N0=[0.1,0.11]
###r1,r2,alpha11,alpha12,alpha21,alpha22
params = (0.1,0.1,1,2,1,2)
modelSim=spint.odeint(func=paperSim,y0=N0,t=times,args=params)
simDF=df.DataFrame({"t":times,"normal":modelSim[:,0],"tumor":modelSim[:,1]})
case2=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
case2.draw()

### Case3 alpha12<alpha11 and alpha22<alpha21
N0=[0.1,0.11]
###r1,r2,alpha11,alpha12,alpha21,alpha22
params = (0.1,0.1,2,1,2,1)
modelSim=spint.odeint(func=paperSim,y0=N0,t=times,args=params)
simDF=df.DataFrame({"t":times,"normal":modelSim[:,0],"tumor":modelSim[:,1]})
case3=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
case3.draw()

### Case 4 alpha11=alpha12 and alpha22=alpha21
N0=[0.1,0.11]
###r1,r2,alpha11,alpha12,alpha21,alpha22
params = (0.1,0.1,2,1,1,2)
modelSim=spint.odeint(func=paperSim,y0=N0,t=times,args=params)
simDF=df.DataFrame({"t":times,"normal":modelSim[:,0],"tumor":modelSim[:,1]})
case4=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
case4.draw()

### Case 5 alpha12>alpha11 and alpha22<alpha21
N0=[0.1,0.11]
###r1,r2,alpha11,alpha12,alpha21,alpha22
params = (0.1,0.1,1,2,2,1)
modelSim=spint.odeint(func=paperSim,y0=N0,t=times,args=params)
simDF=df.DataFrame({"t":times,"normal":modelSim[:,0],"tumor":modelSim[:,1]})
case5=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
case5.draw()
