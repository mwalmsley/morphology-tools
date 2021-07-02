import random
import logging
import json

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic

from modAL.models import ActiveLearner, BayesianOptimizer
# from modAL.models import CommitteeRegressor
# from modAL.disagreement import vote_entropy_sampling, max_std_sampling
from modAL.acquisition import max_EI

import shared


def teach_optimizer(optimizer, X_pool, y_pool, retrain_size):
    query_indices, _ = optimizer.query(X_pool, n_instances=retrain_size)
    X_acquired = X_pool[query_indices]
    y_acquired = y_pool[query_indices]
    optimizer.teach(X_acquired, y_acquired)  # bootstrap=True
    return X_acquired, y_acquired  # for analysis


# i.e. exploration mode
def max_uncertainty_query_strategy(modal_learner, X, n_instances=1):
    _, stds = modal_learner.predict(X, return_std=True)
    top_indices = np.argsort(stds)[::-1][:n_instances]  # ::-1 to sort high to low
    return top_indices, X[top_indices].squeeze()


# i.e. exploitation mode
def max_value_query_strategy(modal_learner, X, n_instances=1):
    preds, _ = modal_learner.predict(X, return_std=True)
    top_indices = np.argsort(preds)[::-1][:n_instances]  # ::-1 to sort high to low
    return top_indices, X[top_indices].squeeze()


# TODO increasing batch size for constant fit time?

def benchmark_gp(n_components=10, n_iterations=10, training_size=10, retrain_size=10, retrain_batches=29):

    # max_galaxies = 1000
    max_galaxies = 60714
    # max_galaxies = 40672  # featured, face-on, good ellipse measurement
    # 40921 in decals/cnn/irregular after cuts, 40672 with ellipses due to nans (same morph. cuts)
    # dataset_name='decals'
    dataset_name ='gz2'

    method = 'ellipse'
    # method = 'cnn'

    # anomalies = 'mergers'
    # anomalies = 'featured'
    anomalies = 'odd'
    # anomalies = 'rings'
    # anomalies = 'ring_responses'
    # anomalies = 'irregular'

    if dataset_name == 'gz2':
        features, labels, responses, metadata = shared.load_gz2_data(method=method, anomalies=anomalies, max_galaxies=max_galaxies)
    elif dataset_name == 'decals':
        features, labels, responses, metadata = shared.load_decals_data(method=method, max_galaxies=max_galaxies, anomalies=anomalies)
    elif dataset_name == 'simulated':
        features, labels, responses, metadata = shared.load_simulated_data()
    else:
        raise ValueError(dataset_name)

    print('Labels: \n', pd.value_counts(labels))
    print('Responses: \n', pd.value_counts(responses))
    if max_galaxies is not None:
        if not len(labels) == max_galaxies:
            logging.warning('Expected {} galaxies but only recieved {}'.format(max_galaxies, len(labels)))

    if method == 'cnn':
        print('Applying PCA for embedding')
        save = 'anomaly/figures/{}/gp_pca_variation.png'.format(dataset_name)
        # save = ''
        embed = shared.get_embed(features, n_components=n_components, save=save)
    else:
        embed = features
    del features # TODO being lazy

    embed_subset, responses_subset, labels_subset = embed[:5000], responses[:5000], labels[:5000]
    sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(labels_subset), alpha=.3)
    plt.savefig('anomaly/figures/{}/simulated_embed_first_2_components_labels_{}.png'.format(dataset_name, anomalies))
    plt.close()
    sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(responses_subset), alpha=.3)
    plt.savefig('anomaly/figures/{}/simulated_embed_first_2_components_responses_{}.png'.format(dataset_name, anomalies))
    plt.close()

    all_metrics = []
    for iteration_n in tqdm.tqdm(np.arange(n_iterations)):

        shuffle_indices = np.arange(len(labels))
        random.shuffle(shuffle_indices)  # inplace
        embed = embed[shuffle_indices]
        labels = labels[shuffle_indices]
        responses = responses[shuffle_indices]
        metadata = metadata.iloc[shuffle_indices].reset_index(drop=True)  # hopefully pandas friendly

        # X_train, y_train, X_pool, y_pool, X_test, y_test = split_three_ways(embed, labels, train_size=10, iteration_n=iteration_n)
        # X_train, X_pool, y_train, y_pool = train_test_split(embed, labels, train_size=training_size, iteration_n=iteration_n)
        # print(X_train.shape, X_pool.shape)
        # print(y_train.shape, y_pool.shape)
        # print('Total interesting: {}'.format(np.isclose(y_test, 4).sum()))

        # learners = []
        # n_learners = 5
        # for _ in range(n_learners):
        #     kernel = RBF() + WhiteKernel()  # or matern
        #     gp = GaussianProcessRegressor(kernel=kernel, iteration_n=iteration_n)
        #     learner = ActiveLearner(
        #         estimator=gp,
        #         query_strategy=max_EI,
        #         X_training=X_train,
        #         y_training=y_train
        #     )
        #     learners.append(learner)

        # committee = CommitteeRegressor(
        #     learner_list=learners,
        #     query_strategy=max_std_sampling
        # )


        # kernel = RBF() + WhiteKernel()  # or matern
        kernel = RationalQuadratic() + WhiteKernel()  # or matern
        gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(X_pool[:1000], y_pool[:1000])
        # print(gp.score(X_test, y_test))

        # kernel = RBF() + WhiteKernel()  # or matern
        # gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(X_pool[:1000], y_pool[:1000])
        # print(gp.score(X_test, y_test))

        # kernel = Matern(nu=1.5) + WhiteKernel()  # or matern
        # gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(X_pool[:1000], y_pool[:1000])
        # print(gp.score(X_test, y_test))

        # kernel = Matern(nu=2.5) + WhiteKernel()  # or matern
        # gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(X_pool[:1000], y_pool[:1000])
        # print(gp.score(X_test, y_test))

        # exit()

        learner = BayesianOptimizer(
            estimator=gp,
            query_strategy=max_EI
        )

        # learner = BayesianOptimizer(
        #     estimator=gp,
        #     query_strategy=max_uncertainty_query_strategy
        # )

        # learner = ActiveLearner(
        #     estimator=gp,
        #     query_strategy=,
        #     X_training=X_train,
        #     y_training=y_train
        # )

        acquired_samples = []
        for batch_n in range(retrain_batches):
            # TODO possibly enforce a hypercube or similarly spread first selection
            # if (batch_n < 5) or (batch_n % 2 == 0):  # if early or even batch_n
            #     # print('explore')
            #     learner.query_strategy = max_uncertainty_query_strategy
            # else:
            #     # print('exploit')
            #     learner.query_strategy = max_value_query_strategy
            

            X_acquired, _ = teach_optimizer(learner, embed, responses, retrain_size)  # trained on responses TODO
            acquired_samples.append(X_acquired)  # viz and count only
            labelled_samples = (1 + batch_n) * retrain_size
            # print('Labelled samples: {}'.format(labelled_samples))

            # preds = committee.predict(X_test)
            # metrics = get_metrics(preds, X_test, y_test)
            preds = learner.predict(embed)
            metrics = shared.get_metrics(preds, labels)  # measured on the labels TODO
            # metrics['score'] = learner.estimator.score(embed, labels)  # doesn't seem to work right - should be responses not labels
            metrics['score'] = explained_variance_score(preds, responses)

            metrics['labelled_samples'] = labelled_samples
            metrics['iteration_n'] = iteration_n
            # print(metrics['labelled_samples'], metrics['accuracy_50'])
            all_metrics.append(metrics)


            # special metrics for fig 5 in astronomaly paper
            if (dataset_name == 'gz2') and (labelled_samples == 200):
                print('Calculating fig 5 metrics')
                sorted_labels = labels[np.argsort(preds)][::-1]

                experiment_name = 'ellipse_loch_{}_{}'.format(np.random.randint(10000), iteration_n)
                shared.get_metrics_like_fig_5(sorted_labels, method, dataset_name, 'gp', experiment_name)

            if batch_n == retrain_batches - 1:

                # embed_subset, responses_subset, labels_subset, preds_subset = embed[:5000], responses[:5000], labels[:5000], preds[:5000]

                fig, ax = plt.subplots()
                ax.scatter(embed[:, 0], embed[:, 1], alpha=.06, s=1.)
                ax.axis('off')
                fig.tight_layout()
                fig.savefig('anomaly/figures/{}/embed_first_2_components_dist_{}.png'.format(dataset_name, anomalies))
                plt.close()
                # sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(preds_subset), alpha=.3)
                fig, ax = plt.subplots()
                print(preds[:30])
                # print((np.clip(preds, 1., 3.)-1)/2.)
                ax.scatter(embed[:, 0], embed[:, 1], c=(np.clip(preds, 1., 3.)-1)/2., alpha=.06, s=1)
                ax.axis('off')
                fig.tight_layout()
                fig.savefig('anomaly/figures/{}/embed_first_2_components_final_preds_{}.png'.format(dataset_name, anomalies))
                plt.close()


                # if (dataset_name == 'gz2') or (dataset_name == 'decals'):
                #     save_loc = '/home/walml/repos/astronomaly/comparison/results/{}/top_12_galaxies_it{}.png'.format(dataset_name, iteration_n)
                #     shared.save_top_galaxies(preds, metadata, save_loc)


        ## only useful for d = 5 and below
        # acquired_samples = np.stack(acquired_samples, axis=0)  # batch, row, feature
        # fig, ax = plt.subplots()
        # reshaped_samples = acquired_samples.reshape((acquired_samples.shape[0] * acquired_samples.shape[1], acquired_samples.shape[2]))
        # colors = np.concatenate([np.ones(retrain_size) * n / retrain_batches for n in range(retrain_batches)])
        # ax.scatter(reshaped_samples[:, 0], reshaped_samples[:, 1], c=colors, alpha=.5, marker='+')
        # ax.set_xlim([-8, 8])
        # ax.set_ylim([-8, 8])
        # # fig.colorbar()
        # fig.tight_layout()
        # fig.savefig('acquired_points_gp_{}.png'.format(iteration_n))



    df_loc = 'anomaly/results/{}/gp_latest_metrics_{}.csv'.format(dataset_name, method)
    df = pd.DataFrame(data=all_metrics)
    df.to_csv(df_loc, index=False)

    df_loc = 'anomaly/results/{}/gp_latest_metrics_{}.csv'.format(dataset_name, method)
    df = pd.read_csv(df_loc)

    save_loc =  'anomaly/results/{}/gp_metrics_total{}_batch{}_d{}_it{}_{}_{}.png'.format(
        dataset_name,
        len(labels),  # num galaxies
        retrain_size,
        n_components,
        n_iterations,
        anomalies,
        method)
    
    shared.visualise_metrics(df, save_loc, total_anomalies=labels.sum())


if __name__ == '__main__':

    benchmark_gp(n_iterations=10, n_components=10)
