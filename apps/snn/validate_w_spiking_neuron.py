# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:11:47 2021

@author: dell
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from mnn_core.maf import *


class InteNFire():
    def __init__(self, num_neurons = 10):
        self.L = 1/20 #ms
        self.Vth = 20
        self.Vres = 0        
        self.Tref = 5 #ms
        self.Vspk = 50
        self.dt = 1e-1 #integration time step (ms)
        self.num_neurons = num_neurons
        #self.T = T    
        self.we = 0.1
        self.wi = 0.4
        self.ei_balance = 0.5 #fraction of total current variance due to inhibitory inputs
        self.inh_curr_mean = 1 #fix the magnitude of the mean inhibition current
        
        self.maf = MomentActivation()
        
        #self.T = 10e3 #ms 
    
    def input_spk_time(self, spk_mean, spk_var):
        '''
            generate gamma distributed spike time
            isi_mean = shape*scale (unit: kHz)
            isi_var = shape*scale^2 (unit: kHz)
        '''
        
        isi_mean = 1/spk_mean
        isi_var = np.power(isi_mean,3)*spk_var
        
        scale = isi_var/isi_mean
        shape = isi_mean*isi_mean/isi_var
        
        num_spikes = int(self.T/isi_mean)*5
        num_samples = num_spikes*self.num_neurons
        
        isi = np.random.gamma(shape, scale, num_samples)
        isi = isi.reshape((self.num_neurons, num_spikes ))
        
        spk_time = np.cumsum(isi, axis=1)
        
        return spk_time
        
    # def input_synaptic_current(self, t, spk_time):
    #     '''Convert input spike time to post-synaptic current'''        
    #     indx1 = t < spk_time
    #     indx2 = t-self.dt < spk_time
    #     num_spks = np.sum(indx1 ^ indx2, axis = 1) #^ = xor
        
    #     return num_spks*self.we
    
    def input_ei_current(self, t):
        '''Convert intput spike (sparse matrix) to post-synaptic current'''
        current = self.we*self.exc_input[:,t] - self.wi*self.inh_input[:,t]        
        return current.toarray().flatten()
        
    
    def input_spike_sparse(self, spk_mean, spk_var):
        
        scale = spk_var/spk_mean/spk_mean
        shape = spk_mean/spk_var
        
        num_spikes = int(self.T*spk_mean)*5
        num_samples = num_spikes*self.num_neurons
        
        isi = np.random.gamma(shape, scale, num_samples)
        isi = isi.reshape((self.num_neurons, num_spikes ))        
        
        spk_time = np.cumsum(isi, axis=1)
        
        #need to do a safety check to make sure that spk_time[:,-1] > self.T for all neurons
        if np.sum(spk_time[:,-1] < self.T):
            print('Warning: not enough spikes!')
        
        spk_time = np.floor( spk_time/self.dt).flatten()
        neuron_index = np.tile( np.arange(self.num_neurons) , (num_spikes,1)).T.flatten()        
        dat = np.ones(num_samples)
        
        spk_mat = sp.sparse.coo_matrix( (dat, (neuron_index, spk_time)), shape = (self.num_neurons, int(np.max(spk_time))+1 ) ).tocsc() #compressed column format
        
        
        spk_mat = spk_mat[:,:int(self.T/self.dt)]
        
        # spk_time = 0
        # neuron_index = np.arange(self.num_neurons)
        # R = [] #row index for neurons
        # C = [] #column index for time
        
        # while True:            #this is too slow!
        #     spk_time += np.random.gamma(shape, scale, self.num_neurons)            
        #     valid_entry = spk_time < self.T #spike time does not exceed simulation time            
        #     if np.sum(valid_entry)==0:                
        #         break
        #     else:
        #         C = np.append(C, np.floor(spk_time[valid_entry]/self.dt) )
        #         R = np.append(R, neuron_index[valid_entry])

        # dat = np.ones(R.size)            
        # spk_mat = sp.sparse.coo_matrix( (dat, (R,C)), shape = (self.num_neurons, int(self.T/self.dt)) ).tocsc() #compressed column format
        return spk_mat
        
    
    def input_gaussian_current(self, mean, std, corr = None):
        ''' Generate gaussian input current '''
        if corr is None:
            input_current = np.random.randn(self.num_neurons)
            input_current = input_current*std*np.sqrt(self.dt) + mean*self.dt
            #input_current = input_current
        else:
            N = len(mean)
            cov = corr*std.reshape(N,1)*std.reshape(1,N)
            input_current = np.random.multivariate_normal(mean*self.dt, cov*self.dt)
            
        return input_current
    
    def run(self, T = None, input_mean = 1, input_std = 1, input_corr = None, ntrials = None, input_type = 'gaussian', record_v = False, show_message = False):
        '''Simulate integrate and fire neurons'''
        
        if T:
            self.T = T            
        else:
            # use adaptive time step based on firing rate predicted by m.a.f.                        
            # only supports scalar input_mean and input_std
            maf_u = self.maf.mean(np.array([input_mean]),np.array([input_std]))
            self.T = min(10e3, 100/maf_u) #T = desired number of spikes / mean firing rate
        
        num_timesteps = int(self.T/self.dt)
        
        tref = np.zeros(self.num_neurons) #tracker for refractory period
        v = np.random.rand(self.num_neurons)*self.Vth #initial voltage
        #v = np.zeros(self.num_neurons)
        
        SpkTime = [[] for i in range(self.num_neurons)]
        t = np.arange(0, self.T , self.dt)
        if record_v:
            V = np.zeros( (self.num_neurons, num_timesteps) )
        else:            
            V = None
                
        mean, std, rho = input_mean, input_std, input_corr # input current stats (not firing rate, it is current)
        if input_type == 'spike':
            #fix mean_inh and std_inh
            inh_var = self.ei_balance*np.power(std/self.wi,2)
            exc_var = (1-self.ei_balance)*np.power(std/self.we,2)
            
            exc_mean = (mean + self.inh_curr_mean)/self.we
            inh_mean = self.inh_curr_mean/self.wi
            
            self.exc_input = self.input_spike_sparse(exc_mean, exc_var)
            self.inh_input = self.input_spike_sparse(inh_mean, inh_var)
        
        start_time = time.time()
        for i in range(num_timesteps):
            
            #evolve forward in time
            if input_type == 'gaussian':
                input_current = self.input_gaussian_current(mean, std)
            elif input_type == 'bivariate_gaussian':
                corr = np.eye(2) + rho*(1-np.eye(2))
                cov = corr*std[:2].reshape(2,1)*std[:2].reshape(1,2)
                input_current = np.random.multivariate_normal(mean[:2]*self.dt, cov*self.dt, size = ntrials) #output is ntrials-by-2
                input_current = input_current.flatten()
                #corr = np.kron( np.eye( int(mean.size/2) ), corr) #create block diagonal matrix
                #input_current = self.input_gaussian_current(mean, std, corr = corr)
                #don't do this too slow.
                # generate bivariate gaussian of shape (2, num of samples) then reshape it
                
                
            elif input_type == 'multivariate_gaussian':
                corr = np.eye(self.num_neurons) + rho*(1-np.eye(self.num_neurons))
                input_current = self.input_gaussian_current(mean, std, corr = corr)
            elif input_type == 'spike':
                #input_current = self.input_synaptic_current(i*self.dt, spk_time)
                input_current = self.input_ei_current(i)                   
                
            v += -v*self.L*self.dt + input_current
            
            #check state
            is_ref = (tref > 0.0) & (tref < self.Tref)
            is_spike = (v > self.Vth) & ~is_ref
            is_sub = ~(is_ref | is_spike) 
            
            v[is_spike] = self.Vspk            
            v[is_ref] = self.Vres
            
            #update refractory period timer
            tref[is_sub] = 0.0
            tref[is_ref | is_spike] += self.dt
            
            
            if record_v:
                V[:,i] = v            
            
            for k in range(self.num_neurons):
                if is_spike[k]:
                    SpkTime[k].append(i*self.dt)
            
            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
                
        return SpkTime, V, t
    
    def empirical_maf(self, SpkTime):
        '''Turns out it's a bad idea to calulate spk stats with isi'''
        # isi = []
        # for spk_time in SpkTime:
        #     spk_time = np.array(spk_time)
        #     spk_time = spk_time[ spk_time > 500] #remove burn-in time; unit: ms
        #     if spk_time.size > 1:
        #         isi.extend( list(np.diff(spk_time)) )
            
        # if len(isi)>30:
        #     mean_isi = np.mean(isi)
        #     var_isi = np.var(isi)
        #     mu = 1/mean_isi        
        #     sig = np.sqrt( np.power(mu,3)*var_isi )
        # else:
        spike_count = [len(spk_time) for spk_time in SpkTime]            
        mu = np.mean(spike_count)/self.T
        sig = np.sqrt(np.var(spike_count)/self.T)
        
        
        return mu, sig

def input_output_analysis_corr(mean1 = 1, std1 = 1, mean2 = 1, std2 = 1, ntrials = 1000):
    inf = InteNFire(num_neurons = 2*ntrials)
    u = np.array([mean1,mean2])    
    s = np.array([std1,std2])
    rho = np.linspace(-1,1,11)
    
    #calculate analytical result
    maf_u = inf.maf.mean(u,s)
    maf_s, _ = inf.maf.std(u,s)
    maf_chi = inf.maf.chi(u,s)
    maf_rho = maf_chi[0]*maf_chi[1]*rho
    T = min(10e3, 100/min(maf_u)) # adaptive simulation time
    print('Using simulation time (ms):',T)
    #simulation result of IF neuron
    u = np.tile(u,ntrials)
    s = np.tile(s,ntrials)
    
    output_rho = rho.copy()    
    for i in range(len(rho)):
        #spk_count = np.zeros((ntrials,2))
        SpkTime, _, _ = inf.run(T = T, input_mean = u, input_std = s, input_corr = rho[i], input_type = 'bivariate_gaussian', ntrials = ntrials)        
        spk_count = np.array([ len(k) for k in SpkTime]).reshape((ntrials,2))
        print('Progress: i={}/{}'.format(i,len(rho)))
        output_rho[i] = np.corrcoef( spk_count[:,0], spk_count[:,1] )[0,1]
    
    
    return rho, output_rho, maf_rho
            
def batch_analysis_corr():
    '''Fix mean2=std2=1 and vary mean1 and st1'''
    #u = np.array([0, 0.5, 1]) #each run takes about 3.5 min
    #s = np.array([0.5, 3, 10])
    u = np.linspace(0,2,11)
    s = np.linspace(0,5,11)
    
    rho_out = np.zeros((11,len(u),len(s)))
    rho_maf = np.zeros((11,len(u),len(s)))
    
    start_time = time.time()
    
    for i in range(len(u)):
        for j in range(len(s)):
            rho_in, rho_out[:,i,j], rho_maf[:,i,j] = input_output_analysis_corr(mean1 = u[i], std1 = s[j], ntrials = 1000)
    
            print('Time elapsed: {} min'.format( (-start_time + time.time())/60 ) )    
    #plt.plot(rho_in, rho_out, '.')
    #plt.plot(rho_in, rho_maf)
    np.save('validate_corr_4',{'rho_in':rho_in,'rho_out':rho_out,'rho_maf':rho_maf,'u':u,'s':s})
    return rho_in, rho_out, rho_maf, u, s 

def batch_analysis_corr2():
    '''Let mean1=mean2 and std1=std2, vary both.'''
    u = np.linspace(0,2,11)
    s = np.linspace(0,2,11)
    
    rho_out = np.zeros((11,len(u),len(s)))
    rho_maf = np.zeros((11,len(u),len(s)))
    
    start_time = time.time()
    
    for i in range(len(u)):
        for j in range(len(s)):
            rho_in, rho_out[:,i,j], rho_maf[:,i,j] = input_output_analysis_corr(mean1 = u[i], mean2 = u[i], std1=s[j], std2 = s[j], ntrials = 1000)
    
            print('Time elapsed: {} min'.format( (-start_time + time.time())/60 ) )    
    #plt.plot(rho_in, rho_out, '.')
    #plt.plot(rho_in, rho_maf)
    np.save('validate_corr_5',{'rho_in':rho_in,'rho_out':rho_out,'rho_maf':rho_maf,'u':u,'s':s})
    return rho_in, rho_out, rho_maf, u, s         
    
def input_output_anlaysis(input_type):
    inf = InteNFire(num_neurons = 1000) #time unit: ms        
    #N = 31
    #u = np.linspace(-0.5,2.5,N)
    N = 51
    u = np.linspace(-0.5,2.5,N)
    s = np.array([1,2])#np.ones(N)*1.5
    
    emp_u = np.zeros((N,s.size))
    emp_s = np.zeros((N,s.size))
    
    start_time = time.time()
    for i in range(N):
        for j in range(s.size):
            SpkTime, _, _ = inf.run(T = None, input_mean = u[i], input_std = s[j], input_type = input_type, show_message = False)
            emp_u[i,j], emp_s[i,j] = inf.empirical_maf(SpkTime)    
                    
        progress = (i+1)/N*100
        elapsed_time = (time.time()-start_time)/60
        print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ))
        
    
    #maf = MomentActivation()
    
    maf_u = np.zeros((N,s.size))
    maf_s = np.zeros((N,s.size))
    
    for j in range(s.size):
        maf_u[:,j] = inf.maf.mean(u,s[j]*np.ones(N))
        maf_s[:,j], _ = inf.maf.std(u,s[j]*np.ones(N))
    
    return emp_u, emp_s, maf_u, maf_s, u, s
    

def simple_demo(input_type):
    inf = InteNFire( num_neurons = 100) #time unit: ms    
    
    SpkTime, V, t = inf.run(T = 1e3,input_type = input_type, record_v = True, show_message = True)
    plt.plot(t,V[0,:])
    plt.plot(SpkTime[0],[51]*len(SpkTime[0]),'.')        
    return inf

def simple_demo_two_neurons():
    inf = InteNFire( num_neurons = 1000) #time unit: ms    
    u = np.array([1,1])
    s = np.array([1,1])
    SpkTime, V, t = inf.run(T = 1e3, input_mean = u, input_std = s, input_corr = -0.1, input_type = 'bivariate_gaussian', record_v = True, show_message = True)
    plt.plot(t,V[0,:])
    plt.plot(t,V[1,:])
        

if __name__=='__main__':
    #input_rho, output_rho, maf_rho = input_output_analysis_corr(mean1 = 0.6, mean2 = 0.6)
    rho_in, rho_out, rho_maf, u, s = batch_analysis_corr2()
    #simple_demo_two_neurons()
    #simple_demo(input_type = 'gaussian' )
    #out = input_output_anlaysis(input_type = 'spike')
    #inf = simple_demo(input_type = 'spike' )
    #runfile('./apps/snn/validate_w_spiking_neuron.py', wdir='./')
    
    #np.save('validate_w_spk_neuron',out)
        