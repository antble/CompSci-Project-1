import numpy as np

def Franke(x1, x2):
    a = 0.75*np.exp(-((9*x1-2)**2)/4-((9*x2-2)**2)/4)
    b = 0.75*np.exp(-((9*x1+1)**2)/49-((9*x2+1)**2)/10)
    c = 0.5*np.exp(-((9*x1-7)**2)/4-((9*x2-3)**2)/4)
    d = 0.2*np.exp(-((9*x1-4)**2)-((9*x2-7)**2))
    return a+b+c-d

#if __name__ == '__main__':
#    #Franke()