"""

read "ring" as "expert says it's a ring"

Loss function for network, p observed data given variable of interest data
p(mobile|f(x)) / N =  p(ring) log p^hat(ring) + p(~ring) log p^hat(~ring)

Normally:
p^hat_ring = f(x), use softmax to ensure [0, 1] interval
p(ring) = 1 if labelled ring, 0 otherwise

Here, uncertain labels:
p(ring) = p(ring|mobile votes), need to estimate. Nice separable problem (and useful in of itself).


p(ring|mobile votes) = p(mobile votes|ring) p(ring) / [p(mobile votes|ring)p(ring) + p(mobile votes|~ring)p(~ring)]

p(ring), p(~ring) are the base rates of expert labels for gz mobile galaxies
p(mobile|ring) is the histogram of mobile responses for expert-labelled rings

But these all implicity assume they were voted on by gz mobile. Explicitly:

p(ring|mobile votes, voted) = p(mobile votes|ring, voted) p(ring, voted) / [p(mobile votes|ring, voted)p(ring, voted) + p(mobile votes|~ring, voted)p(~ring, voted)]

Most galaxies would not be selected, so classifier trained to estimate p(ring|mobile votes, voted) would be miscalibrated

Need a correction to reduce estimate according to the rate of expert rings

Bayes rule to calibrate vs. base rates
p(ring|f(x)) = p(f(x)|ring)p(ring)/ [ p(f(x)|ring)p(ring) + p(f(x)|p(~ring)) ]


Now p(ring) and p(f(x)|p(ring)) are the histograms/rates for expert-labelled rings in random (from the selected subset) galaxies
Potentially tedious to estimate - may need to find say 100 rings -> few thousand random galaxies, or 30 rings -> 1000 galaxies. 
"""

from collections import Counter

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import tensorflow_probability as tfp


"""Utilities for setting up experiment"""

def select_for_gz_mobile(df, num_galaxies=1000):
    assert len(df) > num_galaxies
    df_sorted = df.sort_values('rare-features_ring_fraction', ascending=False)
    return df_sorted[:num_galaxies], df_sorted[num_galaxies:]


def get_fake_mobile_votes(gz_ring_vote_fractions):
    return scipy.stats.binom(n=10, p=gz_ring_vote_fractions).rvs(len(gz_ring_vote_fractions))

"""Loss functions"""

# def binomial_loss(y_true, y_pred):  # y_true should be the counts, y_pred should be the predicted probabilities
#     return -tfp.distributions.Binomial(total_count=10, probs=y_pred).log_prob(y_true)  # important minus sign

def dirichlet_multinomial_loss(y_true, y_pred):
    # expects y_pred to be [alpha, beta], each from e.g. 1 to 101 if using Lambda layer
    votes = tf.concat([y_true, 10-y_true], axis=1)  # need to mirror the "no" vote as could in principle be many answers
    return -tfp.distributions.DirichletMultinomial(total_count=10, concentration=y_pred).log_prob(votes)

"""Convert prediction to scores"""

# def concentrations_to_mode_prob(concentrations):
#     return (concentrations[:, 0]-1)/(concentrations[:, 1]+concentrations[:, 0]-2)

def concentrations_to_mean_prob(concentrations):
#     https://en.wikipedia.org/wiki/Beta_distribution
    return 1/(1+(concentrations[:, 1]/concentrations[:, 0]))

# def concentrations_to_mean_prob_tfp(concentrations):
#     # exactly as concentrations_to_mean_prob but via tfp (no benefit)
# #     WHY WHY WHY is concentration0 beta and concentration1 alpha...
#     beta = tfp.distributions.Beta(concentration1=concentrations[:,0], concentration0=concentrations[:, 1])
#     return beta.mean()

# note, more scores than outputs when using this
def preds_to_sampled_vote(concentrations, num_samples=100):
#     WHY WHY WHY is concentration0 beta and concentration1 alpha...
    beta = tfp.distributions.Beta(concentration1=concentrations[:,0], concentration0=concentrations[:, 1])
    return beta.sample(num_samples).numpy().flatten()


"""Bin scores with percentile bins. Not needed for fixed bins as easy"""


def get_percentile_bins(x, num_bins=10):
    percentiles_to_check = [n * (100/num_bins) for n in range(0, num_bins+1)]

    # 11 percentiles, forming 10 bins
    percentiles = np.percentile(x, percentiles_to_check)  # percentiles of all predictions
    percentiles[-1] = percentiles[-1] + 1e-8  # deals with the awkwards == max case
    
    bins = np.array([(low_p, high_p) for (low_p, high_p) in zip(percentiles[:-1], percentiles[1:])])
    bin_centers = np.array([low_p + ((high_p - low_p)/2) for (low_p, high_p) in zip(percentiles[:-1], percentiles[1:])])
    return percentiles, bins, bin_centers


def get_percentile_histogram(x, percentiles, normalize=True):
    # digitize gives index such that x is between bins[i-1] < x < bins[i]
    # values outside the range snap to the closest bin (i.e. 0 or len(bins))
    bin_of_each_x = np.digitize(x, percentiles)
    x_per_bin_counter = Counter(bin_of_each_x)
#     rings_per_bin_counter  # will have indices from 1 to 11 (as x == min bin gives 1, not 0)
    assert x_per_bin_counter[len(percentiles)+1] == 0
    
    num_bins = len(percentiles)-1
    x_per_bin = np.zeros(num_bins)  # 10 
    for bin_index in range(num_bins): # 0 to 10
        x_per_bin[bin_index] = x_per_bin_counter[bin_index+1]  # indexed via 1 to 11
        
    assert x_per_bin.sum() == len(x)

    if normalize:
        return x_per_bin / len(x)
    else:
        return x_per_bin  # counts of values from percentile[i] to percentile[i+1]. Will be one less then len(percentiles)        

"""Combine pred->score and score histograms to get ring/not ring rates"""

def get_percentile_ring_rates(all_preds, random_expert_ring_preds, random_expert_not_ring_preds, pred_to_score_func):
#     better for score of single predicted p, which won't cover whole domain
    
    all_outputs = pred_to_score_func(all_preds)
    random_expert_ring_outputs = pred_to_score_func(random_expert_ring_preds)
    random_expert_not_ring_outputs = pred_to_score_func(random_expert_not_ring_preds)
    
    percentiles, _, bin_centers = get_percentile_bins(all_outputs, num_bins=10)

    ring_rates = get_percentile_histogram(random_expert_ring_outputs.squeeze(), percentiles, normalize=True)
    not_ring_rates = get_percentile_histogram(random_expert_not_ring_outputs.squeeze(), percentiles, normalize=True)
    return percentiles, bin_centers, ring_rates, not_ring_rates


# all_preds not needed
def get_fixed_bin_ring_rates(random_expert_ring_preds, random_expert_not_ring_preds, pred_to_score_func=preds_to_sampled_vote):
#     better for sampled outputs of p, 0-1 domain enforced
    random_expert_ring_outputs = pred_to_score_func(random_expert_ring_preds)
    random_expert_not_ring_outputs = pred_to_score_func(random_expert_not_ring_preds)
    
    _, bin_edges = np.histogram(np.linspace(0., 1.), bins=30)
#     _ = plt.hist(random_expert_ring_outputs, alpha=.5, density=True, bins=bin_edges)
#     _ = plt.hist(random_expert_not_ring_outputs, alpha=.5, density=True, bins=bin_edges)

    ring_counts, _ = np.histogram(random_expert_ring_outputs, bins=bin_edges)
    ring_rates = ring_counts / len(random_expert_ring_outputs)
    
    not_ring_counts, _ = np.histogram(random_expert_not_ring_outputs, bins=bin_edges)
    not_ring_rates = not_ring_counts / len(random_expert_not_ring_outputs)
    
    assert np.allclose(ring_rates.sum(), 1)
    assert np.allclose(not_ring_rates.sum(), 1)
    
    bin_centers = (bin_edges + (bin_edges[1] - bin_edges[0])/2)[:-1]

#     print(len(ring_rates), len(not_ring_rates))
    return bin_edges, bin_centers, ring_rates, not_ring_rates


"""Get value from histogram"""

def empirical_prob_from_histogram(values_to_query, rates, bin_edges):
    bin_edges = bin_edges.copy()
    bin_edges[-1]= bin_edges[-1]+1e-8  # if values_to_query includes exactly the max bin edge
    # rates from 0 to 10, corresponding to frequencies between each (0-11) bin edge 
    assert all(values_to_query <= bin_edges.max())  # no rate above the highest bin edge, bins must cover whole range
    bin_indices = np.digitize(values_to_query, bin_edges)  # should be 10 indices
#     print(bin_indices.min(), bin_indices.max())
    return np.array([rates[bin_index-1] for bin_index in bin_indices])


"""Apply Bayes to convert scores to prob(ring)"""

def scale_probabity_to_random_sample_via_hists(scores, ring_rates, not_ring_rates, p_ring, bin_edges):
    p_score_given_ring = empirical_prob_from_histogram(scores, ring_rates, bin_edges)
    p_score_given_not_ring = empirical_prob_from_histogram(scores, not_ring_rates, bin_edges)
    return scale_probability_by_bayes(p_score_given_ring, p_score_given_not_ring, p_ring)

def scale_probabity_to_random_sample_via_func(scores, ring_rate_func, not_ring_rate_func, p_ring):
    p_score_given_ring = ring_rate_func(scores)
    p_score_given_not_ring = not_ring_rate_func(scores)
    return scale_probability_by_bayes(p_score_given_ring, p_score_given_not_ring, p_ring)
    
def scale_probability_by_bayes(p_score_given_ring, p_score_given_not_ring, p_ring):
    return p_score_given_ring * p_ring / (p_score_given_ring * p_ring + p_score_given_not_ring * (1-p_ring))
