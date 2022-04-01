import sys

'''
WRITE YOUR CODE BELOW.
'''

from numpy import zeros, float32

#  pgmpy

import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import random

#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function 
    
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")
    
    BayesNet.add_edge("faulty alarm","alarm")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("temperature","faulty gauge")
    
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    
    cpd_f_a = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    cpd_t = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    
    cpd_f_g_given_t = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], [ 0.05, 0.8]], evidence=['temperature'], evidence_card=[2])
    
    cpd_g_given_f_g_and_t = TabularCPD('gauge', 2, values=[[0.95, 0.05, 0.20, 0.80],[0.05, 0.95, 0.80, 0.20]], evidence=['faulty gauge', 'temperature'], evidence_card=[2, 2])
    cpd_a_given_g_and_f_a = TabularCPD('alarm', 2, values=[[0.90, 0.55, 0.10, 0.45],[0.10, 0.45, 0.90, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])
    
    bayes_net.add_cpds(cpd_f_a, cpd_t, cpd_f_g_given_t, cpd_g_given_f_g_and_t, cpd_a_given_g_and_f_a)
    
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values

    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty gauge':0,'faulty alarm' :0}, joint=False)
    temp_prob = conditional_prob['temperature'].values
    
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out

    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    BayesNet.add_edge("A","CvA")

    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    
    cpd_AvB_given_A_and_B = TabularCPD('AvB', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10], [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['A', 'B'], evidence_card=[4, 4])
    cpd_BvC_given_B_and_C = TabularCPD('BvC', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10], [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_CvA_given_C_and_A = TabularCPD('CvA', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10], [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['C', 'A'], evidence_card=[4, 4])
    
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB_given_A_and_B, cpd_BvC_given_B_and_C, cpd_CvA_given_C_and_A)
   
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """    
    # TODO: finish this function
    
    if len(initial_state) < 6:
        initial_state = [random.randint(0,3), random.randint(0,3), random.randint(0,3), 0, random.randint(0,2), 2]
    
    A_cpd = bayes_net.get_cpds('A')      
    team_table = A_cpd.values
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    
    chosen_node_index = random.randint(0,5)
    
    while chosen_node_index == 3 or chosen_node_index == 5:
        chosen_node_index = random.randint(0,5)
    
    if chosen_node_index == 0:
        prob_num_0 = team_table[0] * match_table[initial_state[3], 0,initial_state[1]] * match_table[initial_state[5], initial_state[2], 0]
        prob_num_1 = team_table[1] * match_table[initial_state[3], 1,initial_state[1]] * match_table[initial_state[5], initial_state[2], 1]
        prob_num_2 = team_table[2] * match_table[initial_state[3], 2,initial_state[1]] * match_table[initial_state[5], initial_state[2], 2]
        prob_num_3 = team_table[3] * match_table[initial_state[3], 3,initial_state[1]] * match_table[initial_state[5], initial_state[2], 3]
        choice_prob = (prob_num_0, prob_num_1, prob_num_2, prob_num_3)
        initial_state[0] = random.choices([0, 1, 2, 3], choice_prob, k = 1)[0]
    
    elif chosen_node_index == 1:
        prob_num_0 = team_table[0] * match_table[initial_state[3], initial_state[0], 0] * match_table[initial_state[4], 0, initial_state[2]]
        prob_num_1 = team_table[1] * match_table[initial_state[3], initial_state[0], 1] * match_table[initial_state[4], 1, initial_state[2]]
        prob_num_2 = team_table[2] * match_table[initial_state[3], initial_state[0], 2] * match_table[initial_state[4], 2, initial_state[2]]
        prob_num_3 = team_table[3] * match_table[initial_state[3], initial_state[0], 3] * match_table[initial_state[4], 3, initial_state[2]]
        choice_prob = (prob_num_0, prob_num_1, prob_num_2, prob_num_3)
        initial_state[1] = random.choices([0, 1, 2, 3], choice_prob, k = 1)[0]
    
    elif chosen_node_index == 2:
        prob_num_0 = team_table[0] * match_table[initial_state[5], 0,initial_state[0]] * match_table[initial_state[4], initial_state[1], 0]
        prob_num_1 = team_table[1] * match_table[initial_state[5], 1,initial_state[0]] * match_table[initial_state[4], initial_state[1], 1]
        prob_num_2 = team_table[2] * match_table[initial_state[5], 2,initial_state[0]] * match_table[initial_state[4], initial_state[1], 2]
        prob_num_3 = team_table[3] * match_table[initial_state[5], 3,initial_state[0]] * match_table[initial_state[4], initial_state[1], 3]
        choice_prob = (prob_num_0, prob_num_1, prob_num_2, prob_num_3)
        initial_state[2] = random.choices([0, 1, 2, 3], choice_prob, k = 1)[0]
    
    #elif chosen_node_index == 3:
    #    choice_prob = (match_table[0, initial_state[0], initial_state[1]], match_table[1, initial_state[0], initial_state[1]], match_table[2, initial_state[0], initial_state[1]])
    #    initial_state[3] = random.choices([0, 1, 2], choice_prob, k = 1)[0]
    
    elif chosen_node_index == 4:
        choice_prob = (match_table[0, initial_state[1], initial_state[2]], match_table[1, initial_state[1], initial_state[2]], match_table[2, initial_state[1], initial_state[2]])
        initial_state[4] = random.choices([0, 1, 2], choice_prob, k = 1)[0]
    
    #else:
    #    choice_prob = (match_table[0, initial_state[2], initial_state[0]], match_table[1, initial_state[2], initial_state[0]], match_table[2, initial_state[2], initial_state[0]])
    #    initial_state[5] = random.choices([0, 1, 2], choice_prob, k = 1)[0]
    
    sample = tuple(initial_state)
    
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    if len(initial_state) < 6:
        initial_state = [random.randint(0,3), random.randint(0,3), random.randint(0,3), 0, random.randint(0,2), 2]
    
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    
    final_state = [random.randint(0,3), random.randint(0,3), random.randint(0,3), 0, random.randint(0,2), 2]
    
    prod_prob_initial = team_table[initial_state[0]]*team_table[initial_state[1]]*team_table[initial_state[2]]*match_table[initial_state[3], initial_state[0], initial_state[1]]*match_table[initial_state[4], initial_state[1], initial_state[2]]*match_table[initial_state[5], initial_state[2], initial_state[0]]
    
    prod_prob_final = team_table[final_state[0]]*team_table[final_state[1]]*team_table[final_state[2]]*match_table[final_state[3], final_state[0], final_state[1]]*match_table[final_state[4], final_state[1], final_state[2]]*match_table[final_state[5], final_state[2], final_state[0]]
    
    if prod_prob_final > prod_prob_initial:
        sample = tuple(final_state)
        return sample
    
    else:
        if random.uniform(0,1) < prod_prob_final/prod_prob_initial:
            sample = tuple(final_state)
            return sample
        else:
            sample = tuple(initial_state)
            return sample

def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    
    Delta = 0.001
    N = 100
    
    test = initial_state
    match_res_pred = [0, 0, 0]
    prev_prob_dist = None
    prob_dist = [0, 0, 0]
    difference = [0, 0, 0]
    rep_ctr = 0
    
    iter_count = 1000000
    escape_var = 1000000
    
    while escape_var != 0:
        
        test = list(Gibbs_sampler(bayes_net, test))
        
        match_res_pred[int(test[4])] += 1
        
        for i in range(0, 3):
            prob_dist[i] = float(match_res_pred[i])/(iter_count - escape_var + 1)
            if prev_prob_dist is not None:
                difference[i] = abs(prob_dist[i] - prev_prob_dist[i])
        
        if prev_prob_dist is not None:
            if (difference[0] < Delta and difference[1] < Delta and difference[2] < Delta) and prev_prob_dist != prob_dist:
                rep_ctr += 1
            
            else:
                rep_ctr = 0
        
        if rep_ctr >= N:
            Gibbs_convergence = prob_dist
            Gibbs_count = iter_count - escape_var + 1
            break
        
        prev_prob_dist = [0, 0, 0]
        for i in range(0, 3):
            prev_prob_dist[i] = prob_dist[i]
        
        #print()
        #print(prev_prob_dist)
        #print()
        
        escape_var = escape_var - 1
    
    test = initial_state
    match_res_pred = [0, 0, 0]
    prev_prob_dist = None
    prob_dist = [0, 0, 0]
    difference = [0, 0, 0]
    rep_ctr = 0
    
    iter_count = 1000000
    escape_var = 1000000
    
    while escape_var != 0:
        
        old_test_val = test
        
        test = list(MH_sampler(bayes_net, test))
        
        if test is old_test_val:
            MH_rejection_count += 1
        
        match_res_pred[int(test[4])] += 1
        
        #print()
        #print("debug A")
        #print(prob_dist)
        #print(prev_prob_dist)
        
        for i in range(0, 3):
            prob_dist[i] = float(match_res_pred[i])/float(iter_count - escape_var + 1)
            if prev_prob_dist is not None:
                difference[i] = abs(prob_dist[i] - prev_prob_dist[i])
        
        #print()
        #print("debug B")
        #print(prob_dist)
        #print(prev_prob_dist)
        
        if prev_prob_dist is not None:
            if (difference[0] < Delta and difference[1] < Delta and difference[2] < Delta) and prev_prob_dist != prob_dist:
                rep_ctr += 1
            
            else:
                rep_ctr = 0
        
        if rep_ctr >= N:
            MH_convergence = prob_dist
            MH_count = iter_count - escape_var + 1
            break
        
        prev_prob_dist = [0, 0, 0]
        for i in range(0, 3):
            prev_prob_dist[i] = prob_dist[i]
        
        #print()
        #print(prev_prob_dist)
        #print()
        #print(rep_ctr)
        
        escape_var = escape_var - 1
    
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    
    Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(get_game_network(),[])
    if Gibbs_count < MH_count:
        choice = 0
        factor = float(MH_count) / float(Gibbs_count)
    else:
        choice = 1
        factor = float(Gibbs_count) / float(MH_count)
    
    #print(options[choice]) #
    #print(factor) #
    #print(Gibbs_count)
    #print(MH_count)
    
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Sambhav Mattoo"
    
#sampling_question()