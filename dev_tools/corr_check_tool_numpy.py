# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 21:01:40 2020
Gradcheck for corr map using a simple 2d example
@author: dell
"""

from mnn_core.maf import *
from mnn_core.mnn_pytorch import *

maf = MomentActivation()

def gradcheck_numpy(inputs):
    
    r = inputs[2]#np.random.rand(N)*2-1 #scalar random corr
    
    u = inputs[0]#np.random.randn(N,2) #mean
    s = inputs[1]#np.random.rand(N,2)+1 #std
    
     
    #base forward
    _ = maf.mean(u,s)
    _ = maf.std(u,s)
    chi = maf.chi(u,s)
    
    y = chi[:,0]*chi[:,1]*r
    
    #perturbed forward
    u_new = u.copy()
    du = 1e-5
    u_new[:,0] += du
    _ = maf.mean(u_new,s)
    _ = maf.std(u_new,s)
    chi_new = maf.chi(u_new,s)
    y_new = chi_new[:,0]*chi_new[:,1]*r
    
    #numeric_grad
    numeric_grad = 2*(y_new - y)/du
    
    #analytic grad
    _ = maf.mean(u,s)
    _ = maf.std(u,s)
    _ = maf.chi(u,s)
    _ = maf.grad_mean(u,s)
    
    chi_grad_u, chi_grad_s = maf.grad_chi(u,s)
    
    
    analytic_grad = 2*chi_grad_u[:,0]*chi[:,1]*r
    
        
    return numeric_grad, analytic_grad

def gradcheck_mnn_corr(inputs):
    #repeat the same test with maf from mnn_corr to make sure maf is correctly implemented
    
    r = inputs[2]#np.random.rand(N)*2-1 #scalar random corr
    
    u = inputs[0]#np.random.randn(N,2) #mean
    s = inputs[1]#np.random.rand(N,2)+1 #std
    
     
    #base forward
    mean_out = mnn_core_func.forward_fast_mean(u, s)
    std_out = mnn_core_func.forward_fast_std(u, s, mean_out)
    chi_out = mnn_core_func.forward_fast_chi(u, s , mean_out, std_out)
    
    y = chi_out[:,0]*chi_out[:,1]*r
    
    #perturbed forward
    u_new = u.copy()
    du = 1e-5
    u_new[:,0] += du
    mean_out_new = mnn_core_func.forward_fast_mean(u_new, s)
    std_out_new = mnn_core_func.forward_fast_std(u_new, s, mean_out_new)
    chi_out_new = mnn_core_func.forward_fast_chi(u_new, s , mean_out_new, std_out_new)
    y_new = chi_out_new[:,0]*chi_out_new[:,1]*r
    
    #numeric_grad
    numeric_grad = 2*(y_new - y)/du
    
    #analytic grad
    chi_grad_mean, chi_grad_std = mnn_core_func.backward_fast_chi(u, s, mean_out, chi_out)
        
    analytic_grad = 2*chi_grad_mean[:,0]*chi_out[:,1]*r
    
        
    return numeric_grad, analytic_grad
        
    
if __name__=='__main__':
    
    N = 3
    r = np.random.rand(N)*2-1 #scalar random corr    
    u = np.random.randn(N,2) #mean
    s = np.random.rand(N,2)+1 #std    
    
    
    numeric_grad, analytic_grad = gradcheck_numpy((u,s,r))    
    print('Numeric grad using maf:\n',numeric_grad)
    print('Analytic grad using maf:\n',analytic_grad)
    
    numeric_grad, analytic_grad = gradcheck_mnn_corr((u,s,r))    
    print('Numeric grad using mnn_corr_func:\n',numeric_grad)
    print('Analytic grad using mnn_corr_func:\n',analytic_grad)
    
    #rel_err = np.abs((analytic_grad - numeric_grad)/numeric_grad)
    #plt.hist(rel_err)