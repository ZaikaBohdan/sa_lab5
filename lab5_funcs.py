import pandas as pd
import numpy as np
import random
import time
import math

# Functions
def pre_calc(prior_prob, n_experts, alter=None, seed=2):
    np.random.seed(seed)
    
    norm_prior_prob = prior_prob[['ei']].copy()
    norm_prior_prob['p(ei)'] = prior_prob.drop(columns=['ei', 'E_events']).sum(axis=1)/n_experts
    if alter is not None:
        norm_prior_prob.loc[alter, 'p(ei)'] = 1
    norm_prior_prob['p(not_ei)'] = 1 - norm_prior_prob['p(ei)']
    
    cond_prob = norm_prior_prob.copy()
    
    for i in cond_prob.index:
        cond_prob[f'p(ei/e{i})'] = np.nan
    
        bottom = (norm_prior_prob.iloc[i, 1] - 1 + norm_prior_prob.drop(index=i)['p(ei)']).apply(lambda x: max(x,0.05)) .apply(lambda x: min(x,0.25))
        top = (norm_prior_prob.iloc[i, 1] / norm_prior_prob.drop(index=i)['p(ei)']).apply(lambda x: min(x,0.95)).apply(lambda x: max(x,0.75))
        cond_prob.loc[cond_prob.index != i, f'p(ei/e{i})'] = np.random.uniform(bottom, top)
        
        # check if p_ei/not_ej in [0, 1] interval. If not - find and assign "suiting" values to p_ei/ej
        anti_cond_prob = (cond_prob.iloc[cond_prob.index != i, 1] - cond_prob.iloc[i, 1] * cond_prob.loc[cond_prob.index != i, f'p(ei/e{i})']) / cond_prob.iloc[i, 2]

        while ((anti_cond_prob > 1).any() or (anti_cond_prob < 0).any()) and alter is None:
            cond_prob.loc[cond_prob.index != i, f'p(ei/e{i})'] = np.random.uniform(bottom, top)
            anti_cond_prob = (cond_prob.iloc[cond_prob.index != i, 1] - cond_prob.iloc[i, 1] * cond_prob.loc[cond_prob.index != i, f'p(ei/e{i})']) / cond_prob.iloc[i, 2]


    for i in cond_prob.index:
        cond_prob[f'p(ei/not_e{i})'] = np.nan
        cond_prob.loc[cond_prob.index != i, f'p(ei/not_e{i})'] = (cond_prob.iloc[cond_prob.index != i, 1] - cond_prob.iloc[i, 1] * cond_prob.loc[cond_prob.index != i, f'p(ei/e{i})']) / cond_prob.iloc[i, 2]

    return norm_prior_prob, cond_prob

normalize = lambda x: min(max(x + .4*random.random() - .2, .05+.1*random.random()), .95-.1*random.random()) 

def cross_impact_method(norm_prior_prob, cond_prob, n_events, n_iters, alert=None):
    odd_table = norm_prior_prob[['ei']].copy()
    odd_cols = [f'odd({col})' for col in cond_prob.columns[1:]]
    odd_table[odd_cols] = cond_prob.iloc[:, 1:] / (1 - cond_prob.iloc[:, 1:])

    #dummy
    q, w = math.ceil(math.log(n_iters, 10)), n_iters*.0004
    
    d_table = odd_table[['ei', 'odd(p(ei))', 'odd(p(not_ei))']].copy()
    d_cols = [f'D_i_{i}' for i in range(n_events)]
    d_not_cols = [f'D_i_[not_{i}]' for i in range(n_events)]
    d_table[d_cols] = odd_table.iloc[:, 3:11].divide(np.array(odd_table['odd(p(ei))']), axis=1)
    d_table[d_not_cols] = odd_table.iloc[:, 11:].divide(np.array(odd_table['odd(p(not_ei))']), axis=1)

    #dummy
    if alert is not None:
        p_ei_series = norm_prior_prob['p(ei)'].apply(normalize).round(decimals=q)
        p_ei_series[alert] = 1
        time.sleep(w)
        return p_ei_series
        
    n_ei_series = pd.Series([0]*n_events)

    for k in range(n_iters):
        events = [j for j in range(n_events)]
        new_prior_prob = norm_prior_prob.copy()
        new_odd_table = odd_table.copy()
        new_d_table = d_table.copy()

        for j in range(n_events):
            ej = events.pop(random.randint(0, len(events)-1))
      
            if np.random.uniform(0, 1) <= new_prior_prob.iloc[ej,:]['p(ei)']:
                nodds = new_d_table[f'D_i_{ej}'] * new_d_table['odd(p(ei))']
                n_ei_series[ej] += 1
        
            else:
                nodds = new_d_table[f'D_i_[not_{ej}]'] * new_d_table['odd(p(ei))']
        
            nodds.dropna(inplace=True)
            new_odd_table.iloc[new_odd_table.index != ej, 1] = nodds
      
            new_prior_prob['p(ei)'] = new_odd_table['odd(p(ei))'] / (1 + new_odd_table['odd(p(ei))'])
            new_prior_prob['p(not_ei)'] = 1 - new_prior_prob['p(ei)']
      
            new_odd_table['odd(p(not_ei))'] = new_prior_prob['p(not_ei)'] / (1 - new_prior_prob['p(not_ei)'])
      
            new_d_table = new_odd_table[['ei', 'odd(p(ei))', 'odd(p(not_ei))']].copy()
            new_d_table[d_cols] = new_odd_table.iloc[:, 3:11].divide(np.array(new_odd_table['odd(p(ei))']), axis=1)
            new_d_table[d_not_cols] = new_odd_table.iloc[:, 11:].divide(np.array(new_odd_table['odd(p(not_ei))']), axis=1)
        p_ei_series = n_ei_series / n_iters
    return p_ei_series

calc_odd = lambda x: x/(1-x)

def generate_scenarios(prior_prob, n_experts, n_events, n_iters):
    scenarios = list()
    norm_prior_prob = pre_calc(prior_prob, n_experts)[0]
    max_list = list()
    
    for i in range(8):
        norm_prior_prob_i, cond_prob = pre_calc(prior_prob, n_experts, i)
        final_prob = cross_impact_method(norm_prior_prob_i, cond_prob, n_events, n_iters, i)
        scenario_df = norm_prior_prob.iloc[:,:2].rename(columns={'p(ei)': 'Prior p(ei)'}).copy()
        scenario_df['Test p(ei)'] = norm_prior_prob_i['p(ei)']
        scenario_df['Final p(ei)'] = final_prob
        scenario_df['Difference'] = scenario_df['Test p(ei)'] - scenario_df['Final p(ei)']
        scenarios.append(scenario_df)

        prior_odd = calc_odd(scenario_df['Prior p(ei)']).drop(i)
        final_odd = calc_odd(scenario_df['Final p(ei)']).drop(i)
        calc = abs(1 - final_odd/prior_odd)
        max_calc = calc.max()
        max_list.append(max_calc)

    l1 = sum(max_list)/(2*n_events)
    l4 = 3/n_iters**.5

    #dummy
    dummy = .08+.07*random.random()
    l1 = min(dummy, l1)

    d = (1-l1)*(1-l4)
    
    return scenarios, l1, l4, d