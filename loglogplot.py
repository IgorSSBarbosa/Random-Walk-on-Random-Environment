import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# taking absolute value of S_n, sorting the samples, calculate mean and median.
def treat_this_data(A):
    absA = np.abs(A)
    sortedAbs = np.sort(absA)
    meanA = absA.mean(1)
    median = np.quantile(absA, 0.5 , 1)
    return (absA,sortedAbs,meanA, median)


# calculating the interval of confidence
def confidence_level(q,sortedAbs, alpha=0.01): # it gives a confifence interval centered in quantile q
    numb_columns,numb_rows = sortedAbs.shape
    # numb_columns is the range of simulations done
    # numb_rows is the number of simulations done

    lowerboundindex = [int(np.floor(q*(numb_rows - np.sqrt(numb_rows)*norm.isf(alpha)))) for i in range(numb_columns)]
    upperboundindex = [int(np.ceil(q*(numb_rows + np.sqrt(numb_rows)*norm.isf(alpha)))) for i in range(numb_columns)]
    
    lowerboundheight = sortedAbs[np.arange(numb_columns),lowerboundindex]
    upperboundheight = sortedAbs[np.arange(numb_columns),upperboundindex]

    quantile_q = sortedAbs[np.arange(numb_columns), int( numb_rows*q ) ]
    
    errorbar = [np.abs(lowerboundheight-quantile_q), np.abs(upperboundheight-quantile_q)]
    return errorbar


def domain(k1,k2):
    predomain = range(k2-k1+1)              # create a list of int [0 , 1 , ... , k2-k1]
    dom = [int(x+k1) for x in predomain]    # shift that list to get [k1, k1+1, ... , k2]

    return dom


def linear_regression(dom,out,q=None,errorbar=None):
    poly_coef = np.polyfit(dom, out, 1)
    poly_out = [poly_coef[1] + poly_coef[0]*x for x in dom]

    # print the results in a graphic
    if q is not None and errorbar is not None: # when quantile and errorbar are given it's possibleto calulate an errorbar
        plt.plot(dom, poly_out,label='y={:.3}x+{:.2}, quantile{:.2}'.format(poly_coef[0],poly_coef[1],q))
        plt.errorbar(dom,out,yerr=errorbar, label = 'quantile {:.2}'.format(q),capsize=4, marker = 'o')
    else: 
        plt.plot(dom, poly_out,label='y={:.3}x+{:.2}, mean'.format(poly_coef[0],poly_coef[1]))
    
    return poly_coef # returning the values of linear coeficcients


def log_log_plot(q,dom, A, alpha=0.01):
    absA, sortedAbs, meanA, median = treat_this_data(A)

    logAbs    = np.log(absA)/np.log(2)
    logquantile = np.quantile(logAbs, q, 1)
    logmean   = np.log(meanA)/np.log(2)
    logmedian = np.log(median)/np.log(2)

    errorbar = confidence_level(q,sortedAbs,alpha)
    errorbar_median = confidence_level(0.5, sortedAbs,alpha)

    quantile_coef = linear_regression(dom,logquantile,q,errorbar)
    median_coef = linear_regression(dom,logmedian,0.5,errorbar_median)
    mean_coef = linear_regression(dom,logmean)

    print(" ---- diffusion constant is approximally %s using quantile %s ----" % (quantile_coef[0],q))
    print(" ---- diffusion constant is approximally %s median ----" % (median_coef[0]))
    print(" ---- diffusion constant is approximally %s mean ----" % (mean_coef[0]))
    
    plt.legend(fontsize=12)
    plt.show()  