import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,quantile_test

# taking absolute value of S_n, sorting the samples, calculate mean and median.
def treat_this_data(A):
    absA = np.abs(A)
    sortedAbs = np.sort(absA)
    meanA = absA.mean(1)
    median = np.quantile(absA,0.5,1)
    return (absA,sortedAbs,meanA, median)

def expand_matrix(k1,k2,A):
    m, n = A.shape
    assert m == k2 - k1 + 1, "Matrix A's row count must be equal to k2 - k1 + 1"
    print('number of simulations =',n)
    print('In red is the mean of each simulation size')
    print('In orange is the median of each simulation size')
    result = []
    for i in range(n):
        for j in range(m):
            k = k1 + j
            result.append([A[j, i], k])
    
    return np.array(result)
# Extract for plotting the histogram
def histogram(k1,k2,A):
    absA, sortedAbs, meanA, median = treat_this_data(A)

    result = expand_matrix(k1,k2,sortedAbs)
    values = result[:, 0]
    ks = result[:, 1]

    dom = range(k1,k2+1)
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(ks,values, alpha=0.1, color='teal')
    plt.scatter(dom,meanA, color='red')
    plt.scatter(dom,median,color='orange')
    plt.xlabel('k')
    plt.ylabel('A[i,j]')
    plt.title('Scatter plot of A[i,j] values vs k')
    plt.grid(True)
    plt.show()


# calculating the interval of confidence

def mult_ci(q,sortedAbs, alpha=0.01): # it gives a confifence interval centered in quantile q
    numb_columns,numb_rows = sortedAbs.shape
    # numb_columns is the range of simulations done
    # numb_rows is the number of simulations done
    ci  = np.zeros((2,numb_columns))
    for i in range(numb_columns):
        res = quantile_test(sortedAbs[i], q=q, p=q)
        ci[0][i],ci[1][i]  = res.confidence_interval(confidence_level=1-alpha)
    return ci


def linear_regression(dom,out,fig,ax,pos_x=0,pos_y=0,q=None,errorbar=None):
    poly_coef = np.polyfit(dom, out, 1)
    poly_out = [poly_coef[1] + poly_coef[0]*x for x in dom]
    # print the results in a graphic
    if q is not None and errorbar is not None: # when quantile and errorbar are given it's possibleto calulate an errorbar
        ax[pos_x,pos_y].plot(dom, poly_out,label='y={:.3}x+{:.2}, quantile{:.2}'.format(poly_coef[0],poly_coef[1],q))
        ax[pos_x,pos_y].legend()
        ax[pos_x,pos_y].errorbar(dom,out,yerr=errorbar, label = 'quantile {:.2}'.format(q),capsize=4, marker = 'o',)
        ax[pos_x,pos_y].legend()
    else:
        ax[pos_x,pos_y].plot(dom, poly_out,label='y={:.3}x+{:.2}, mean'.format(poly_coef[0],poly_coef[1]),)
        ax[pos_x,pos_y].plot(dom,out,label='mean samples',marker = 'o',)
        ax[pos_x,pos_y].legend(fontsize=12)
    return poly_coef # returning the values of linear coeficcients


def log_log_plot(q,dom, A, alpha=0.01):
    absA, sortedAbs, meanA, median = treat_this_data(A)

    logAbs    = np.log(absA)/np.log(2)
    logquantile = np.quantile(logAbs, q, 1)
    logmean   = np.log(meanA)/np.log(2)
    logmedian = np.log(median)/np.log(2)

    log_ci = np.log(mult_ci(q,sortedAbs,alpha))/np.log(2)
    log_ci_median = np.log(mult_ci(0.5, sortedAbs,alpha))/np.log(2)

    log_errorbar = [logquantile- log_ci[0],log_ci[1]-logquantile]
    log_errorbar_median = [logmedian -log_ci_median[0], log_ci_median[1]-logmedian]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    quantile_coef = linear_regression(dom,logquantile,fig,ax,pos_x=0,pos_y=0,q=q,errorbar=log_errorbar)
    median_coef = linear_regression(dom,logmedian,fig,ax,pos_x=0,pos_y=1,q=0.5,errorbar=log_errorbar_median)
    mean_coef = linear_regression(dom,logmean,fig,ax,pos_x=1,pos_y=0)

    plt.tight_layout()
    plt.show()  

    print(" ---- diffusion constant is approximally %s using quantile %s ----" % (quantile_coef[0],q))
    print(" ---- diffusion constant is approximally %s median ----" % (median_coef[0]))
    print(" ---- diffusion constant is approximally %s mean ----" % (mean_coef[0])) 