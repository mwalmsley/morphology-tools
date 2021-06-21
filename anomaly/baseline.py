import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import explained_variance_score

from astronomaly.anomaly_detection import human_loop_learning

import shared


def joint_predict(rescaled_anomaly_scores, regressor_scores, distances):
    weighted_preds = human_loop_learning.weight_by_distance(
        distances,
        regressor_scores,
        rescaled_anomaly_scores,
        min_score=0.1, max_score=5., alpha=1.
    )
    plt.hist(distances, label='distances')
    plt.hist(regressor_scores, label='regressor')
    plt.hist(rescaled_anomaly_scores, label='anomaly')
    plt.hist(weighted_preds, label='weighted')
    plt.legend()
    plt.savefig('temp.png')
    plt.close()
    return weighted_preds


def benchmark_default(retrain_size=10, retrain_batches=30):

    # max_galaxies = 10000
    max_galaxies = 60714
    # max_galaxies = 40672

    dataset_name = 'gz2'
    # dataset_name = 'decals'

    method = 'ellipse'
    # method = 'cnn'

    # anomalies = 'mergers'
    # anomalies = 'rings'
    # anomalies = 'ring_responses'
    # anomalies = 'irregular'
    anomalies = 'odd'


    if dataset_name == 'gz2':
        features, labels, responses, metadata = shared.load_gz2_data(method=method, anomalies=anomalies, max_galaxies=max_galaxies)
    elif dataset_name == 'decals':
        features, labels, responses, metadata = shared.load_decals_data(method=method, anomalies=anomalies, max_galaxies=max_galaxies)
    elif dataset_name == 'simulated':
        features, labels, responses, metadata = shared.load_simulated_data()
    else:
        raise ValueError(dataset_name)

    print('Labels: \n', pd.value_counts(labels))
    print('Responses: \n', pd.value_counts(responses))
    if max_galaxies is not None:
        if not len(labels) == max_galaxies:
            logging.warning('Expected {} galaxies but only recieved {}'.format(max_galaxies, len(labels)))

    # always use embed with cnn, 1000 features is silly - probably?
    if method == 'cnn':
        print('Applying PCA for embedding')
        features = shared.get_embed(features, n_components=10, save='')  # optionally compress first with PCA

    #human_loop_learning.py:195
    regressor = RandomForestRegressor(n_estimators=100)

    if dataset_name == 'simulated':
        unsupervised_estimator = LocalOutlierFactor(n_neighbors=100, contamination='auto', novelty=False)
        unsupervised_estimator.fit_predict(features)
        # ScoreConverter has lower_is_weirder=True by default i.e. multiply by -1. This ensures low/weird anomalies get high/recommended scores.
        all_scores = -1 * unsupervised_estimator.negative_outlier_factor_
    elif dataset_name == 'gz2' or dataset_name == 'decals':
        unsupervised_estimator = IsolationForest(n_estimators=100, contamination='auto')
        unsupervised_estimator.fit(features)
        # also -1, lower decision function = more abnormal
        all_scores = -1 * unsupervised_estimator.decision_function(features)
    # paper is not clear what output of iforest is used, but code seems to use decision_function

    rescaled_scores = human_loop_learning.rescale_array(all_scores, new_min=0., new_max=5., convert_integer=False)
    score_indices = np.argsort(rescaled_scores)[::-1]  # high to low

    sorted_X = features[score_indices]
    sorted_labels = labels[score_indices]
    sorted_responses = responses[score_indices]
    anomaly_preds = rescaled_scores[score_indices]
    # very very skewed towards nearly everything being normal
    print(anomaly_preds.min(), anomaly_preds.mean(), anomaly_preds.max())
    print(np.mean(anomaly_preds > 2))

    # shared.visualise_predictions_in_first_two_dims(sorted_X, anomaly_preds, 'comparison/figures/anomaly_preds.png')
    # shared.visualise_predictions_in_first_two_dims(sorted_X, sorted_labels, 'comparison/figures/true_labels.png')

    # no outer random_state loop yet
    all_metrics = []
    for retrain_batch in tqdm.tqdm(range(retrain_batches)):
        labelled_samples = (retrain_batch+1) * retrain_size
        slice_to_label = slice(retrain_batch*retrain_size, (retrain_batch+1)*retrain_size)

        # pretend to label them
        if len(all_metrics) == 0:   # first iteration, initialise
            regressor_X = sorted_X[slice_to_label]
            regressor_y = sorted_responses[slice_to_label]
        else:
            regressor_X = np.concatenate([regressor_X, sorted_X[slice_to_label]], axis=0)
            regressor_y = np.concatenate([regressor_y, sorted_responses[slice_to_label]], axis=0)

        # update regressor
        regressor.fit(regressor_X, regressor_y.squeeze())
        regressor_preds = regressor.predict(sorted_X)
        
        distances = human_loop_learning.get_distances(regressor_X, sorted_X)  # distances of sorted_X to closest (labelled) regressor_X
        # shared.visualise_predictions_in_first_two_dims(sorted_X, distances, 'comparison/figures/distances_labelled_{}.png'.format(labelled_samples))

        joint_preds = joint_predict(anomaly_preds, regressor_preds, distances)

        # TEMP disable joint preds
        # metrics = shared.get_metrics(anomaly_preds, sorted_labels)  # no test set, applied on everything
        metrics = shared.get_metrics(joint_preds, sorted_labels)  # no test set, applied on everything
        metrics['labelled_samples'] = labelled_samples
        metrics['random_state'] = 0
        metrics['score'] = explained_variance_score(regressor_preds, sorted_responses)
        all_metrics.append(metrics)

        # shared.visualise_predictions_in_first_two_dims(sorted_X, regressor_preds, 'comparison/figures/regressor_preds_labelled_{}.png'.format(labelled_samples))
        # shared.visualise_predictions_in_first_two_dims(sorted_X, joint_preds, 'comparison/figures/joint_preds_labelled_{}.png'.format(labelled_samples))
        # shared.visualise_predictions_in_first_two_dims(regressor_X, regressor_y, 'comparison/figures/acquired_labels_{}.png'.format(labelled_samples), xlim=[-7.5, 7.5], ylim=[-7, 8.5])

    df_loc = 'comparison/results/{}/latest_baseline_metrics_{}.csv'.format(dataset_name, method)
    df = pd.DataFrame(all_metrics)
    df.to_csv(df_loc, index=False)

    df_loc = 'comparison/results/{}/latest_baseline_metrics_{}.csv'.format(dataset_name, method)
    df = pd.read_csv(df_loc)

    save_loc = 'comparison/results/{}/baseline_metrics_total{}_batch{}_{}_{}.png'.format(
        dataset_name,
        len(features),  # num galaxies
        retrain_size,
        anomalies,
        method)

    shared.visualise_metrics(df, save_loc, total_anomalies=labels.sum())  # assumes responses, not classes


if __name__ == '__main__':

    benchmark_default()
