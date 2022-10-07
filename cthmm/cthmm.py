'''
This is the main file.
It was developed at Zeitworks Inc, written by Field Cady.

'''

#
# Author: Field Cady, Feb 2022
#

import numpy as np
from scipy.linalg import fractional_matrix_power, expm
from numpy.linalg import inv
import pandas as pd
import time

class BaseCTHMM:
    def __init__(self,
                 # States
                 n_states=None,
                 states=None,
                 # Transitions
                 Q=None,
                 holding_time=None,
                 startprob=None,
                 endprob=None
                ):
        # Either pass in the # of states, or a list of what the states are called
        if (n_states is not None) and (states is not None):
            self.n_states = n_states
            self.states = states
            assert(n_states==len(states))
        elif n_states is not None:
            self.n_states = n_states
            self.states = [s for s in range(n_states)]
        elif states is not None:
            self.n_states = len(states)
            self.states = states
        else: raise Exception('You must specify the number of states or a list of states')
        n_states = self.n_states
        # Either pass in Q or the mean_holding_times, or just a holding_time
        if Q is not None: self.Q = Q
        elif holding_time is not None: self.Q = default_Q(self.n_states, holding_time=holding_time)
        else: raise Exception('You must specify Q or the avg holding time')
        # start/end probs
        if startprob is not None: self.startprob=startprob
        else: self.startprob = np.ones((n_states,))/n_states
        if endprob is not None: self.endprob=endprob
        else: self.endprob = np.ones((n_states,))/n_states
        # Make sure everything is self-consistent
        assert(len(self.startprob)==self.n_states, 'Len of startprob must equal # states')
        assert(len(self.endprob)==self.n_states, 'Len of endprob must equal # states')
    def get_stationary_distribution(self):
        max_holding_time = -1.0/self.Q.diagonal().max()
        LONG_TIME = 1000*max_holding_time
        trans_probs = expm(LONG_TIME*self.Q)
        return trans_probs[0,:].reshape((self.n_states, ))
    def simulate(self, n=100, seed=None, sample_dt = np.random.random):
        '''Return DataFrame giving timestamps, underlying state, and emission at the time.
        '''
        if seed: np.random.seed(seed)
        states = np.array(range(self.n_states))
        t = 0
        probs = self.startprob
        state_vec = 0*states
        time_seq, state_seq, observation_seq = [], [], []
        for i in range(n):
            st = np.random.choice(states, p=probs/sum(probs))
            state_vec *= 0
            state_vec[st] = 1
            observation = self.get_observation(st)
            time_seq.append(t); state_seq.append(st); observation_seq.append(observation)
            dT = sample_dt()#[0]
            t += dT
            diff = expm(dT*self.Q)
            new_probs = np.matmul(state_vec, diff)
            probs = new_probs
        return pd.DataFrame({'time':time_seq, 'state':state_seq, 'emission':observation_seq})
    # Predicting / decoding
    def predict(self, observations, times, algorithm='viterbi'):
        algorithm = algorithm.lower()
        if algorithm=='viterbi':
            observation_probs = self.get_observation_probs(observations)
            return viterbi(observation_probs, self.Q, times, startprob=self.startprob)
        elif algorithm in ['map', 'fb', 'forward-backward', 'forward/backward', 'forward_backward']:
            observation_probs = self.get_observation_probs(observations)
            state_seq, probs_array = forward_backward(
                observation_probs, self.Q, times, startprob=self.startprob, endprob=self.endprob)
            return state_seq
        else: raise NotImplemented("Only 'viterbi' and 'map' algorithms are supported by predict() function")
    def predict_proba(self, observations, times, algorithm='viterbi'):
        observation_probs = self.get_observation_probs(observations)
        state_seq, probs_array = forward_backward(observation_probs, self.Q, times, startprob=self.startprob, endprob=self.endprob)
        return probs_array
    def interpolate(self, observations, times, times_to_interp):
        # Interpolate times later than or equal to times[0]
        observation_probs = self.get_observation_probs(observations)
        state_seq, probs_array = forward_backward(observation_probs, self.Q, times, startprob=self.startprob, endprob=self.endprob)
        guesses = []
        i_known, i_interp = 0, 0
        startprob = probs_array[0].flatten()
        endprob = probs_array[-1].flatten()
        start_time, end_time = times[0], times[len(times)-1]
        while i_interp<len(times_to_interp):
            t_interp = times_to_interp[i_interp]
            if t_interp<start_time:
                # interp backward from first guess
                dT = times[0]-t_interp
                guess_probs = self.interpolate_backward(startprob, dT)
                guesses.append(guess_probs)
                i_interp += 1
            elif t_interp<end_time:
                #print('foo', t_interp, end_time, i_known, times[i_known], times[i_known+1])
                if t_interp<times[i_known]: i_known+=1
                elif times[i_known+1]<t_interp: i_known+=1
                else:
                    # t_interp between times[i_known] and times[i_known+1]
                    dT1 = t_interp-times[i_known]
                    dT2 = times[i_known+1]-t_interp
                    guess_probs1 = self.interpolate_forward(probs_array[i_known].flatten(), dT1)
                    guess_probs2 = self.interpolate_forward(probs_array[i_known+1].flatten(), dT1)
                    guess_probs = (dT2*guess_probs1 + dT1*guess_probs2) / (dT1+dT2)
                    guesses.append(guess_probs)
                    i_interp += 1
            else:
                # interp forward from last guess
                dT = t_interp-end_time
                guess_probs = self.interpolate_backward(endprob, dT)
                guesses.append(guess_probs)
                i_interp += 1
        return np.array(guesses)            
    def interpolate_forward(self, x, dT):
        trans_mat = expm(dT*self.Q)
        return np.matmul(trans_mat, x)
    def interpolate_backward(self, x, dT):
        trans_mat = expm(dT*self.Q.T)
        return np.matmul(trans_mat, x)
    def get_logprob(self, observations, states, times):
        observation_probs = self.get_observation_probs(observations)
        return get_logprob(observation_probs, states, times, Q=self.Q, startprob=self.startprob)
    # Fitting
    def fit(self, observations_times_pairs, fit_startprob=False, max_iter=10, tol=1e-5, verbose=False):
        """
            Fit the model (matrix Q and Pr[obs|state], and optionally the start/end probs) to a
            List of (observations, timestamps) pairs.
        """
        if verbose: print(self)
        for i in range(max_iter):
            if verbose: print('\nIteration', i)
            prev_Q = self.Q
            delt = 0
            self.fit_observation_params(observations_times_pairs, verbose=verbose)
            self.fit_Q(observations_times_pairs, progress=verbose)
            delt += ((self.Q-prev_Q)**2).sum().sum()
            if verbose: print(self)
            if delt<tol:
                if verbose: print('Delta < tolerance.  Converged!')
                break
            if i==max_iter-1: print(f'WARNING: failure to converge.  delta={delt}')
    def fit_Q(self, observations_times_pairs, progress=False):
        n_total_obs = sum(len(obs) for obs, times in observations_times_pairs)
        Q_ = self.Q.copy()
        Q_total_ = 0 * Q_
        n_observations_total_ = 0
        for observations, times in observations_times_pairs:
            observation_probs = self.get_observation_probs(observations)
            #state_seq, state_probs = forward_backward(observation_probs, Q_, times, startprob=self.startprob, endprob=self.endprob)
            state_seq = viterbi(observation_probs, Q_, times, startprob=self.startprob)
            Q_partial_ = fit_Q_1seq(state_seq, times, start_Q=Q_)
            Q_total_ += len(observations)*Q_partial_
            n_observations_total_ += len(observations)
        self.Q = Q_total_ / n_observations_total_
    def fit_observation_params(self, observations_times_pairs, fit_startprob=False, verbose=True, max_iter=5, seed=None, tol=1e-6,):
        """
            Uses Baum-Welch algorithm to fit observation parameters to a collection of observation sequences.
            If optional argument fit_startprob=True (default False) then it will also fit the start/end
            probabilities (which are 1/n_states otherwise).
        """
        # Fits emission probabilities.  Currently does NOT fit the start/end probs
        if verbose: print(f'Starting emission_probs:\n', self.emission_probs)   
        start_time = time.time()
        startprob_ = self.startprob
        endprob_ = self.endprob
        for i in range(max_iter):
            startprob_new_ = 0*self.startprob
            endprob_new_ = 0*self.endprob
            observations_lst, state_prob_arrays_lst = [], []
            for observations, times in observations_times_pairs:
                observation_probs_ = self.get_observation_probs(observations)
                fw_seq_, state_probs_ = forward_backward(observation_probs_, self.Q, times,
                                                               startprob=self.startprob, endprob=endprob_)
                observations_lst.append(observations)
                state_prob_arrays_lst.append(state_probs_)
                startprob_new_ += state_probs_[0,:]
                endprob_new_ += state_probs_[-1,:]
            combo_observations = np.concatenate(observations_lst, axis=0)
            combo_state_probs = np.concatenate(state_prob_arrays_lst, axis=0)
            delt = self.fit_observation_params_mle(combo_observations, combo_state_probs)
            if fit_startprob:
                startprob_ = startprob_new_ / sum(startprob_new_)
                endprob_ = endprob_new_ / sum(endprob_new_)
            if delt < tol: break
            #if i==max_iter-1: print(f'WARNING: max iterations ({max_iter}) exceeded.  failed to converge')
        end_time = time.time()
        if verbose: print(f'Updated emission_probs.  time={round(end_time-start_time)}sec\n', self.emission_probs)
    # To be implemented
    def get_observation_probs(self, observations):
        raise NotImplemented
    def get_observation(self, st):
        raise NotImplemented

class MultinomialCTHMM(BaseCTHMM):
    ''' Observations are multinomial with values ranging from 0 to k-1 '''
    def __init__(self,
                 # States
                 n_states=None,
                 states=None,
                 # Transitions
                 Q=None,
                 holding_time=None,
                 startprob=None,
                 endprob=None,
                 # Emissions
                 n_emissions=None,
                 emission_probs=None,
                 seed=42
                ):
        # Set all the stuff for state(s) ad their transitions
        super().__init__(n_states=n_states, states=states, Q=Q,
                          holding_time=holding_time, startprob=startprob, endprob=endprob)
        # Mutinomial Specific
        if n_emissions is not None: self.n_emissions = n_emissions
        else: self.n_emissions = emission_probs.shape[1]
        if emission_probs is not None:
            n_emissions = emission_probs.shape[1]
            self.emissions = np.array(range(n_emissions))
        if emission_probs is not None: self.emission_probs = emission_probs
        else:
            # Randomly initialize
            n_states, n_emissions = self.n_states, self.n_emissions
            emission_probs_ = np.random.beta(1,1,n_states*n_emissions).reshape((n_states, n_emissions))
            self.emission_probs = emission_probs_/emission_probs_.sum(axis=1).reshape((n_states,1))
    def get_observation_probs(self, observations):
        '''
            Input: list of multinomial observations
            Output: (n_observation, n_states) of how likely each observation O is conditioned on state S
        '''
        n_observations = len(observations)
        observation_probs = np.zeros((n_observations, self.n_states))
        for i in range(n_observations):
            obs = observations[i]
            observation_probs[i,:] = self.emission_probs[:,obs]
        return observation_probs
    def get_observation(self, st):
        "Simulate random observation assuming in particular state"
        return np.random.choice(self.emissions, p=self.emission_probs[st,:])
    def fit_observation_params_mle(self, observations, state_probs):
        '''
            Fit observation params for each state to a collection of observations
            and how likely they were to come from each state.
            Input: list/array of observations (possibly agg across multiple seriess)
                    state_probs: n_obs*n_states array of state probs giving Pr[obs|state]
            Return a delta between old and new params
        '''
        emission_probs_new_ = 0*self.emission_probs
        for i, obs in enumerate(observations):
            emission_probs_new_[:,obs] += state_probs[i,:]
        emission_probs_new = emission_probs_new_ / emission_probs_new_.sum(axis=1).reshape((self.n_states,1))
        delt = ((self.emission_probs-emission_probs_new)**2).sum().sum()
        self.emission_probs = emission_probs_new
        return delt
    def __str__(self):
        return f'** Q:\n{self.Q}\n** Emission probs:\n{self.emission_probs}'


class GaussianCTHMM(BaseCTHMM):
    ''' Observations are multinomial with values ranging from 0 to k-1 '''
    def __init__(self,
                 # States
                 n_states=None,
                 states=None,
                 # Transitions
                 Q=None,
                 holding_time=None,
                 startprob=None,
                 endprob=None,
                 # Emissions
                 means=None,
                 stds=None
                ):
        # Set all the stuff for state(s) ad their transitions
        super().__init__(n_states=n_states, states=states, Q=Q,
                          holding_time=holding_time, startprob=startprob, endprob=endprob)
        # Mutinomial Specific
        if means: self.means=means
        else:
            # Leave everything as None if Means not specified
            self.means=None
            self.stds=None
            return
        if stds: self.stds=stds
    def get_observation_probs(self, observations):
        '''
            Input: list of observations
            Output: (n_observations, n_states) of how likely each observation O is conditioned on state S
        '''
        n_observations = len(observations)
        observation_probs = np.zeros((n_observations, self.n_states))
        for i in range(self.n_states):
            diffs = observations-self.means[i]
            observation_probs[:,i] = np.exp(-1*(diffs/self.stds[i])**2)
        return observation_probs
    def get_observation(self, st):
        "Simulate random observation assuming in particular state"
        return np.random.normal(self.means[st], self.stds[st])
    def fit_observation_params_mle(self, observations, state_probs):
        '''
            Fit observation params for each state to a collection of observations
            and how likely they were to come from each state.
            Input: list/array of observations (possibly agg across multiple seriess)
                    state_probs: n_obs*n_states array of state probs giving Pr[obs|state]
            Return a delta between old and new params
        '''
        #emission_probs_new_ = 0*self.emission_probs
        #for i, obs in enumerate(observations):
        #    emission_probs_new_[:,obs] += state_probs[i,:]
        #emission_probs_new = emission_probs_new_ / emission_probs_new_.sum(axis=1).reshape((self.n_states,1))
        #delt = ((self.emission_probs-emission_probs_new)**2).sum().sum()
        #self.emission_probs = emission_probs_new
        #return delt
    def __str__(self):
        return f'** Means:\n{self.means}\n** Deviations:\n{self.stds}'


#
# Core Decoding Algorithms
#

def viterbi(observation_probs, Q, times, startprob=None, progress=False):
    n_observations, n_states = observation_probs.shape
    # Take logs for numerical stability
    observation_scores = np.log(observation_probs)
    # Extract params and initialize trellis diagrams
    n_steps, n_states = observation_probs.shape
    scores_trellis = np.zeros(observation_probs.shape)
    prev_state_trellis = np.zeros(observation_probs.shape)
    scores_trellis[0,:]=observation_scores[0,:]+np.log(startprob)
    # Populate trellis diagram
    for i in range(1, n_steps):
        dt = times[i]-times[i-1]
        trans_probs = expm(dt*Q)
        trans_scores = np.log(trans_probs)
        if progress and i % 1000 == 0: print(f"Step {i} of {n_steps}")
        for s in range(n_states):
            scores_by_prev_state = scores_trellis[i-1,:]+trans_scores[:,s].T+observation_scores[i,:]
            scores_trellis[i,s] = np.max(scores_by_prev_state)
            prev_state_trellis[i,s] = np.argmax(scores_by_prev_state)
    # Work backward to get best sequence of states
    states_in_reverse = [np.argmax(scores_trellis[-1,:])]
    for i in range(n_steps-1, 0, -1):
        si = states_in_reverse[-1]
        prev_s = prev_state_trellis[i, int(si)]
        states_in_reverse.append(prev_s)
    return np.flip(states_in_reverse).astype(int)

def forward_backward(observation_probs, Q, times, startprob=None, endprob=None, progress=False):
    n_observations, n_states = observation_probs.shape
    # Forward - compute Pr[si | j<i]
    forward_probs = np.zeros((n_observations, n_states))
    for i in range(n_observations):
        if i==0: forward_probs[i,:] = startprob
        else:
            dt = times[i]-times[i-1]
            trans_probs = expm(dt*Q)
            for s in range(n_states):
                forward_probs[i,s] = sum(forward_probs[i-1,:]*observation_probs[i-1,:]*trans_probs[:,s])
            forward_probs[i,:] = forward_probs[i,:] / sum(forward_probs[i,:])
    # Backward - compute Pr[si | j>i]
    backward_probs = np.zeros((n_observations, n_states))
    for i in range(n_observations-1, -1, -1):
        if i==n_observations-1: backward_probs[i,:] = endprob
        else:
            dt = times[i+1]-times[i]
            trans_probs = expm(dt*Q)
            for s in range(n_states):
                backward_probs[i,s] = sum(trans_probs[s,:]*observation_probs[i+1,:]*backward_probs[i+1,:])
            backward_probs[i,:] = backward_probs[i,:] / sum(backward_probs[i,:])
    # Combine
    combined_probs = forward_probs  * observation_probs * backward_probs
    combined_probs = combined_probs / combined_probs.sum(axis=1).reshape((n_observations,1))
    state_seq = combined_probs.argmax(axis=1)
    return state_seq, combined_probs

#
# Core functions for dealing w Q matrix
#

def _holding_times_to_rate_matrix(mean_holding_times):
    '''
    Used when you are too lazy/ignorant to figure out a rate of
    probability flow from all states X-->Y.
    Instead all you know is how long on average each state lasts.
    
    Input: list of avg holding times for each state
    Output: Q matrix with those holding times, where state S flows equally fast to all other states
    '''
    n_states = len(mean_holding_times)
    rates = [1/ht for ht in mean_holding_times]
    diag = np.diag(rates)  # rates along the diagonal
    full = np.ones((n_states, n_states))
    full *= np.array(rates).reshape((n_states, 1))  # each row is copy of the rates
    return diag - full/(n_states-1) + diag/(n_states-1)



def _runs_iter(states, times, avg_holding_times):
    ''' yield start_idx, state S, estimated time in S, next_state (if we can guess it)
    '''
    i = 0
    start_idx = 0
    start_time = times[i]
    cur_state = states[i]
    while i<len(states)-1:
        dT = times[i+1]-times[i]
        next_state = states[i+1]
        time_cutoff = avg_holding_times[cur_state] + avg_holding_times[next_state]
        # Say we are in state S1 and time T1, and next observation is at T2.
        # Let dT = T2-T1 and avgS = avg holding time of state S.  Then
        #   dT << avgS and S2==S1: Probably no state transitions between T1 and T2
        #   dT << avgS and S2!=S1: Probably 1 state transition S1-->S2 between T1 and T2
        #   dT >> avgS: We can't safely assume which transitions happened between T1 and T2
        if next_state==cur_state and dT<time_cutoff:
            # assume next timestamp is from the same run
            pass
        elif next_state==cur_state and dT>=time_cutoff:
            # assume next timestamp is from a new run
            end_time = times[i]
            yield start_idx, cur_state, end_time-start_time, None
            start_idx = i+1
            start_time = times[i+1]
        elif next_state!=cur_state and dT<time_cutoff:
            # assume next timestamp is from the next run
            end_time = times[i] + dT/2
            yield start_idx, cur_state, end_time-start_time, next_state
            start_idx = i+1
            cur_state = next_state
            start_time = times[i+1] - dT/2
        elif next_state!=cur_state and dT>=time_cutoff:
            # assume multiple runs between this and next timestep
            end_time = times[i]
            yield start_idx, cur_state, end_time-start_time, None
            start_idx = i+1
            cur_state = next_state
            start_time = times[i+1]
        i += 1
    end_time = times[i] + avg_holding_times[cur_state]
    yield start_idx, cur_state, end_time-start_time, None

def fit_Q_1seq(state_seq, times, start_Q=None, n_states=None):
    ''' Approx best-fit a Q matrix to states and timestamps
    '''
    if n_states is None: n_states = start_Q.shape[0]
    state_to_durations = [[] for s in range(n_states)]
    transitions = 0.5*np.ones(start_Q.shape)

    avg_holding_times = -1 / start_Q.diagonal()

    for idx, st, ln, next_st, in _runs_iter(state_seq, times, avg_holding_times):
        state_to_durations[st].append(ln)
        if next_st is not None:
            transitions[st, next_st] += 1

    for st in range(n_states): state_to_durations[st] = pd.Series(state_to_durations[st]).mean()

    diag = np.diag(-1 / np.array(state_to_durations))
    nondiag = transitions / transitions.sum(axis=1).reshape((n_states, 1))
    nondiag *= (1 / np.array(state_to_durations)).reshape((n_states, 1))
    Q_ = diag + nondiag
    return Q_

def get_logprob(observation_probs, states, times, Q, startprob=None):
    ''' Return LogProb of a sequence of states and observations
    '''
    n_observations = observation_probs.shape[0]#len(observations)
    n_states = Q.shape[0]
    # Take logs for numerical stability
    if startprob is None: startprob = np.ones(n_states) / n_states
    observation_logprobs = np.log(observation_probs)
    logprob = np.log(startprob[states[0]])
    logprob += observation_probs[states[0], states[0]]
    for i in range(1, n_observations):
        dt = times[i]-times[i-1]
        trans_probs = expm(dt*Q)
        logprob += np.log(trans_probs[states[i-1],states[i]])
        logprob += observation_logprobs[i, states[i]]
    return logprob

def random_Q(n_states):
    ''' Generate random Q matrix for n_states.  Avg holding time ~1 unit of time.
    '''
    X = np.random.beta(1,1,n_states*n_states).reshape((n_states, n_states))
    np.fill_diagonal(X, 0.0)
    diag = -1*np.diag(X.sum(axis=1))
    return X + diag

def default_Q(n_states, holding_time=1.0):
    ''' Generate uniform Q matrix for n_states.  Avg holding time =1 unit of time.
    '''
    X = np.ones((n_states, n_states)) / (n_states-1)
    np.fill_diagonal(X, -1)
    try: holding_time = np.array(holding_time).reshape((n_states,1))
    except: pass
    return X / holding_time

