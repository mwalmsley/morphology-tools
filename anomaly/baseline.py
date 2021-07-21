import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import json

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import explained_variance_score

from astronomaly.anomaly_detection import human_loop_learning

import shared


def joint_predict(rescaled_anomaly_scores, regressor_scores, distances):
    # downweight anomaly scores where the human-regressor has nearby labels (low dist penalty) and low interest (low regressor scores)
    # interestingly, the humans cannot upweight i.e. increase the score of galaxies that isolationforest scored low - they can only decrease others
    weighted_anomaly_scores = human_loop_learning.weight_by_distance(
        distances,
        regressor_scores,
        rescaled_anomaly_scores,
        min_score=0.1, # e1 = 0.1
        max_score=5., #Umax = 5
        alpha=1.  # "alpha has been set to 1"
        # alpha=0.01
    )
    # _, bins = np.histogram(np.linspace(0., 200.))
    # currently, isolation forest predictions are essentially unweighted so active learning is doing nothing
    # plt.hist(distances, bins=bins, label='distances', alpha=.5)   

    # _, bins = np.histogram(np.linspace(0., 6.))
    # plt.hist(regressor_scores, bins=bins, label='regressor', alpha=.5)
    # plt.hist(rescaled_anomaly_scores, bins=bins, label='anomaly', alpha=.5)
    # plt.hist(weighted_anomaly_scores, bins=bins, label='weighted', alpha=.5)

    # score_change = (weighted_anomaly_scores - rescaled_anomaly_scores)/rescaled_anomaly_scores
    # plt.hist(score_change, label='score_change', range=[-1., 0.])

    # plt.legend()
    # plt.savefig('temp.png')
    # plt.close()
    return weighted_anomaly_scores


def benchmark_default(retrain_size=10, retrain_batches=22, run_n=None):
    

    # max_galaxies = 1000
    # max_galaxies = 10000
    # max_galaxies = 60715  # 61578 gz2 kaggle galaxies, a few with nan ellipse features (1 extra nan with my ellipse features)
    # max_galaxies = 40672
    max_galaxies = None

    # dataset_name = 'gz2'
    dataset_name = 'decals'

    method = 'ellipse'
    # method = 'cnn'

    # anomalies = 'mergers'
    # anomalies = 'rings'
    # anomalies = 'ring_responses'
    anomalies = 'irregular'
    # anomalies = 'odd'

    experiment_name = '{}_{}_nofilter_final_{}'.format(method, anomalies, run_n)

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

    # exit()

    if max_galaxies is not None:
        if not len(labels) == max_galaxies:
            logging.warning('Expected {} galaxies but only recieved {}'.format(max_galaxies, len(labels)))

    # always use embed with cnn, 1000 features is silly - probably?
    if method == 'cnn':
        print('Applying PCA for embedding')
        raise NotImplementedError
        # features = shared.get_embed(features, n_components=10, save='')  # optionally compress first with PCA


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

    # sort by isolation forest prediction
    rescaled_scores = human_loop_learning.rescale_array(all_scores, new_min=0., new_max=5., convert_integer=False)

    # print(metadata.columns.values)
    # score_df = pd.DataFrame(data={'rescaled_scores': rescaled_scores, 'all_scores': all_scores, 'objid': metadata['objid']})
    # score_df.to_csv('temp_score_df.csv', index=False)


    score_indices = np.argsort(rescaled_scores)[::-1]  # high to low

    forest_sorted_X = features[score_indices]
    forest_sorted_responses = responses[score_indices]
    forest_sorted_anomaly_preds = rescaled_scores[score_indices]

    # shared.visualise_predictions_in_first_two_dims(sorted_X, anomaly_preds, 'comparison/figures/anomaly_preds.png')
    # shared.visualise_predictions_in_first_two_dims(sorted_X, sorted_labels, 'comparison/figures/true_labels.png')

    # no outer random_state loop yet
    all_metrics = []
    for retrain_batch in tqdm.tqdm(range(retrain_batches)):
        labelled_samples = (retrain_batch+1) * retrain_size
        slice_to_label = slice(retrain_batch*retrain_size, (retrain_batch+1)*retrain_size)

        # pretend to label them
        if len(all_metrics) == 0:
            # first iteration, initialise - regress on this slice
            regressor_X = forest_sorted_X[slice_to_label]
            regressor_y = forest_sorted_responses[slice_to_label]
        else:
            # continuing iteration, regress on this slice plus all previous slices
            regressor_X = np.concatenate([regressor_X, forest_sorted_X[slice_to_label]], axis=0)
            regressor_y = np.concatenate([regressor_y, forest_sorted_responses[slice_to_label]], axis=0)

        # update regressor
        regressor = RandomForestRegressor(n_estimators=100)  # redefine just in case
        regressor.fit(regressor_X, regressor_y.squeeze())
        regressor_preds = regressor.predict(forest_sorted_X)
        
        # expects (labelled points, query points)
        distances = human_loop_learning.get_distances(regressor_X, forest_sorted_X)  # distances of forest_sorted_X to closest (labelled) regressor_X
        # shared.visualise_predictions_in_first_two_dims(sorted_X, distances, 'comparison/figures/distances_labelled_{}.png'.format(labelled_samples))

        # expects (anomaly score, human-estimated regressor score, distances)
        joint_preds = joint_predict(forest_sorted_anomaly_preds, regressor_preds, distances)
        # TEMP disable joint preds?
        # joint_preds = forest_sorted_anomaly_preds

        forest_sorted_labels = labels[score_indices]  # simply to make sure they start off with the same indices as joint preds (i.e. forest-prioritised)
        active_weighted_sorted_labels = forest_sorted_labels[np.argsort(joint_preds)[::-1]]

        metrics = shared.get_metrics(joint_preds, active_weighted_sorted_labels)  # no test set, applied on everything
        metrics['labelled_samples'] = labelled_samples
        metrics['random_state'] = 0
        metrics['score'] = explained_variance_score(regressor_preds, forest_sorted_responses)


        # special metrics for fig 5 in astronomaly paper
        if labelled_samples == 200:
            
            print('Human scores at N = 200: ', pd.value_counts(regressor_y))
            
            print('Calculating fig 5 metrics')

            shared.get_metrics_like_fig_5(active_weighted_sorted_labels, method, dataset_name, 'forest', experiment_name)

            # active_df = pd.DataFrame(data={'rescaled_scores': rescaled_scores[score_indices], 'all_scores': all_scores[score_indices], 'objid': metadata['objid'].values[score_indices], 'joint_preds': joint_preds, 'labels': labels[score_indices], 'active_weighted_sorted_labels': active_weighted_sorted_labels})
            # active_df.to_csv('temp_active_df.csv', index=False)

            predictions_record = {
                'preds_with_labels': joint_preds.astype(float).tolist(),
                'responses': forest_sorted_responses.astype(int).tolist(),
                'labels': forest_sorted_labels.astype(int).tolist(),
                'acquired_features': np.array(regressor_X).tolist(),
                'acquired_labels': np.array(regressor_y).tolist()
            }
            with open('anomaly/results/{}/predictions_forest_{}_{}.json'.format(dataset_name, method, experiment_name), 'w') as f:
                json.dump(predictions_record, f)


        all_metrics.append(metrics)

        # shared.visualise_predictions_in_first_two_dims(sorted_X, regressor_preds, 'comparison/figures/regressor_preds_labelled_{}.png'.format(labelled_samples))
        # shared.visualise_predictions_in_first_two_dims(sorted_X, joint_preds, 'comparison/figures/joint_preds_labelled_{}.png'.format(labelled_samples))
        # shared.visualise_predictions_in_first_two_dims(regressor_X, regressor_y, 'comparison/figures/acquired_labels_{}.png'.format(labelled_samples), xlim=[-7.5, 7.5], ylim=[-7, 8.5])

    # df_loc = 'anomaly/results/{}/latest_baseline_metrics_{}.csv'.format(dataset_name, method)
    # df = pd.DataFrame(all_metrics)
    # df.to_csv(df_loc, index=False)

    # df_loc = 'anomaly/results/{}/latest_baseline_metrics_{}.csv'.format(dataset_name, method)
    # df = pd.read_csv(df_loc)

    # save_loc = 'anomaly/results/{}/baseline_metrics_total{}_batch{}_{}_{}.png'.format(
    #     dataset_name,
    #     len(features),  # num galaxies
    #     retrain_size,
    #     anomalies,
    #     method)

    # shared.visualise_metrics(df, save_loc, total_anomalies=labels.sum())  # assumes responses, not classes


if __name__ == '__main__':

    for run_n in range(15):
        print(run_n)
        benchmark_default(run_n=run_n)
