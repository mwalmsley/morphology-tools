import random
import logging
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('notebook')
import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import IsolationForest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler

from modAL.models import ActiveLearner, BayesianOptimizer
# from modAL.models import CommitteeRegressor
# from modAL.disagreement import vote_entropy_sampling, max_std_sampling
from modAL.acquisition import max_EI

import shared


def teach_optimizer(optimizer, X_pool, y_pool, retrain_size, query_indices=None):
    if query_indices is None:  # otherwise, override with provided indices
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
    max_galaxies = 60715
    # max_galaxies = 40672  # featured, face-on, good ellipse measurement
    # 40921 in decals/cnn/irregular after cuts, 40672 with ellipses due to nans (same morph. cuts)
    # dataset_name='decals'
    dataset_name ='gz2'

    # method = 'ellipse'
    method = 'cnn'

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
        save_variance = 'anomaly/figures/{}/gp_pca_variation.png'.format(dataset_name)
        save_embed = 'anomaly/data/latest_embed.pickle'
        new_embed = True  # may actually need to be true, due to shuffling galaxies - embed won't match up naively
        embed = shared.get_embed(features, n_components=n_components, save_variance=save_variance, save_embed=save_embed, new=new_embed)
        logging.info('Applying zero mean unit variance transform to embed')
        # for ellipses only, apply sklearn StandardScalar i.e. zero mean unit variance transform as per astronomaly
        scl = StandardScaler()
        embed = scl.fit_transform(embed)

    else:
        embed = features
    del features # TODO being lazy

    embed_subset, responses_subset, labels_subset = embed[:5000], responses[:5000], labels[:5000]
    sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(labels_subset), alpha=.3)
    plt.savefig('anomaly/figures/{}/embed_first_2_components_labels_{}.png'.format(dataset_name, anomalies))
    plt.close()
    sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(responses_subset), alpha=.3)
    plt.savefig('anomaly/figures/{}/embed_first_2_components_responses_{}.png'.format(dataset_name, anomalies))
    plt.close()

    all_metrics = []
    for iteration_n in tqdm.tqdm(np.arange(n_iterations)):

        # without the reshuffle of the data and reshuffle of the starting 10, exactly the same galaxies are selected and the variation is completely gone
        # (all failed in fact)
        # with only the reshuffle of the starting 10, the variation is present as normal - so the variation between runs really is the start and not the data shuffle itself

        shuffle_indices = np.arange(len(labels))
        # random.shuffle(shuffle_indices)  # inplace
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
        # gp.fit(embed[:1000], responses[:1000])
        # print(gp.score(embed[-1000:], responses[-1000:]))

        # kernel = Matern(nu=1.5) + WhiteKernel()  # or matern
        # gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(embed[:1000], responses[:1000])
        # print(gp.score(embed[-1000:], responses[-1000:]))

        # kernel = Matern(nu=2.5) + WhiteKernel()  # or matern
        # gp = GaussianProcessRegressor(kernel=kernel, random_state=iteration_n)
        # gp.fit(embed[:1000], responses[:1000])
        # print(gp.score(embed[-1000:], responses[-1000:]))

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
        total_labelled_samples = 0
        for batch_n in range(retrain_batches):

            # TODO possibly enforce a hypercube or similarly spread first selection, or perhaps cluster, or...
            # if (batch_n < 5) or (batch_n % 2 == 0):  # if early or even batch_n
            #     # print('explore')
            #     learner.query_strategy = max_uncertainty_query_strategy
            # else:
            #     # print('exploit')
            #     learner.query_strategy = max_value_query_strategy

            # if batch_n == 0:
                # # use isolationforest to pick the first retrain_size starting examples
                # unsupervised_estimator = IsolationForest(n_estimators=100, contamination='auto')
                # unsupervised_estimator.fit(embed)
                # # lower decision function = more abnormal
                # all_scores = -1 * unsupervised_estimator.decision_function(embed)
                # query_indices = np.argsort(all_scores)[:retrain_size]

                # the same random every iteration, and for gz2 possibly not random unless preshuffled
                # query_indices = np.arange(retrain_size)  

                # # random 10 every time, regardless of dataset overall shuffle
                # query_indices = np.arange(len(embed))
                # random.shuffle(query_indices)
                # query_indices = query_indices[:retrain_size]

            if batch_n == 0:
                 # just the first batch to be random ("random" anyway but this allows bootstrapped performance measurement)
                query_indices = np.arange(len(embed))
                random.shuffle(query_indices)
                retrain_size = 10
                query_indices = query_indices[:retrain_size] 

            # elif batch_n < 40:
            # ultimately this hurt performance and didn't stop occasional failures
            #     retrain_size = 1  # override
            #     # learner.query_strategy = max_uncertainty_query_strategy
            #     query_indices = None
            #     # print('uncertainy sampling')

            else:
                retrain_size = 10  # reset (hacky, lazy)
                query_indices = None
                learner.query_strategy = max_EI
            

            X_acquired, _ = teach_optimizer(learner, embed, responses, retrain_size, query_indices=query_indices)  # trained on responses TODO
            acquired_samples.append(X_acquired)  # viz and count only
            total_labelled_samples += len(X_acquired)
            # print('Labelled samples: {}'.format(labelled_samples))

            # preds = committee.predict(X_test)
            # metrics = get_metrics(preds, X_test, y_test)
            preds = learner.predict(embed)
            metrics = shared.get_metrics(preds, labels)  # measured on the labels TODO
            # metrics['score'] = learner.estimator.score(embed, labels)  # doesn't seem to work right - should be responses not labels
            metrics['score'] = explained_variance_score(preds, responses)

            metrics['labelled_samples'] = total_labelled_samples
            metrics['iteration_n'] = iteration_n
            # print(metrics['total_labelled_samples'], metrics['accuracy_50'])
            all_metrics.append(metrics)


            # print(total_labelled_samples)
            # special metrics for fig 5 in astronomaly paper
            if (dataset_name == 'gz2') and (total_labelled_samples == 200):
                print('Calculating fig 5 metrics')
                sorted_labels = labels[np.argsort(preds)][::-1]

                experiment_name = 'cnn_replication_comp40_{}'.format(iteration_n)
                shared.get_metrics_like_fig_5(sorted_labels, method, dataset_name, 'gp', experiment_name)

            if batch_n == retrain_batches - 1:

                # embed_subset, responses_subset, labels_subset, preds_subset = embed[:5000], responses[:5000], labels[:5000], preds[:5000]

                fig, ax = plt.subplots()
                ax.scatter(embed[:, 0], embed[:, 1], alpha=.06, s=1.)
                ax.set_xlim([-2, 4.2])
                ax.set_ylim([-3, 2])
                # ax.axis('off')
                fig.tight_layout()
                fig.savefig('anomaly/figures/{}/embed_first_2_components_dist_{}_{}.png'.format(dataset_name, anomalies, experiment_name))
                plt.close()
                # sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(preds_subset), alpha=.3)
                fig, ax = plt.subplots()

                # print((np.clip(preds, 1., 3.)-1)/2.)
                ax.scatter(embed[:, 0], embed[:, 1], c=(np.clip(preds, 1., 3.)-1)/2., alpha=.06, s=1)
                ax.set_xlim([-2, 4.2])
                ax.set_ylim([-3, 2])
                # ax.axis('off')
                fig.tight_layout()
                fig.savefig('anomaly/figures/{}/embed_first_2_components_final_preds_{}_{}.png'.format(dataset_name, anomalies, experiment_name))
                plt.close()


                # if (dataset_name == 'gz2') or (dataset_name == 'decals'):
                #     save_loc = '/home/walml/repos/astronomaly/comparison/results/{}/top_12_galaxies_it{}.png'.format(dataset_name, iteration_n)
                #     shared.save_top_galaxies(preds, metadata, save_loc)

                "only works when all batch sizes are equal"
                # acquired_samples = np.stack(acquired_samples, axis=0)  # batch, row, feature
                # with open('anomaly/figures/{}/{}_acquired.pickle'.format(dataset_name, experiment_name), 'wb') as f:
                #     pickle.dump(acquired_samples, f)
                # fig, ax = plt.subplots()
                # # reshape to (row, feature) i.e. drop batch dimension
                # reshaped_samples = acquired_samples.reshape((acquired_samples.shape[0] * acquired_samples.shape[1], acquired_samples.shape[2]))
                # # color first batch by 1, second by...etc
                # colors = np.concatenate([np.ones(retrain_size) * n / retrain_batches for n in range(retrain_batches)])
                # ax.scatter(reshaped_samples[:, 0], reshaped_samples[:, 1], c=colors, alpha=.5, marker='+')
                # ax.set_xlim([-2, 4.2])
                # ax.set_ylim([-3, 2])
                # # fig.colorbar()
                # fig.tight_layout()
                # # purple is early, yellow is late
                # fig.savefig('anomaly/figures/{}/acquired_points_gp_{}_{}.png'.format(dataset_name, anomalies, experiment_name))



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

    benchmark_gp(n_iterations=15, n_components=40, retrain_batches=29)

    # performance is best with 40 components, then 10, then 20. Poor with 200.

    # acquiring the first 40 sequentially (max EI) doesn't help 
    # performance is very poor with max EI and 1 galaxy per batch always - noise helps regularise?

    # initialising with 100 random points hurts/doesn't solve
    # initialising with 10 uncertainty-sampled-1-at-a-time hurts/doesn't solve
    # simply changing the random first point is enough to cause occasional failures
    # the failures are not obviously different when visualised with umap - missing small clusters?
    # I think I will have to call this good enough - 30 runs, 4-5 weaker performance (not complete failures)
    # still averages out well ahead of astronomaly paper

    # from notebook, isolation forest does a fairly bad job at initialising with cnn features
    # (matches up with fairly mediocre performance)

    # notebook has beautiful figures of everything working
    # the umapped regression fitting the anomaly density well 
    # and the acquired points similarly
