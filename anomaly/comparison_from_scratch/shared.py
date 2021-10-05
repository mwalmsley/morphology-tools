import os
import random
import logging
import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
import pandas as pd
import numpy as np
from PIL import Image
from scipy import stats
import tqdm

from sklearn.metrics import recall_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


def load_decals_data(method, anomalies, max_galaxies=None):

    local = os.path.isdir('/Users/walml')

    label_df = get_decals_label_df(anomalies, local)  # no dependence on method or features
    # label_df['objid'] = label_df['iauname']

    if method == 'ellipse':
        # ellipse fitting to galaxies in dr5_volunteer_catalog_internal, not quite as many as auto_posteriors
        feature_df = pd.read_parquet('/Users/walml/repos/morphology-tools/anomaly/data/decals_ellipse_features.parquet')  
        feature_cols = feature_df.columns.values
        feature_df['iauname'] = feature_df.index.astype(str)
        feature_df = feature_df.reset_index(drop=True)
        # or just use df.dropna
        bad_rows = np.any(pd.isna(feature_df.values), axis=1)
        print(f'Bad rows: {bad_rows.sum()}')
        feature_df = feature_df[~bad_rows]

        df = pd.merge(feature_df, label_df, how='inner', on='iauname')
        assert len(feature_df) == len(df)  # all features should have a label


    elif method == 'cnn':
        if local:
            feature_loc = '/Volumes/beta/cnn_features/decals/cnn_features_decals.parquet'
        else:
            feature_loc = 'decals/cnn_features_concat.parquet'
        feature_df = pd.read_parquet(feature_loc)  # features and png_loc
        feature_df['iauname'] = feature_df['iauname'].astype(str)
        # only the DR5 galaxies for now
        feature_df = feature_df[feature_df['iauname'].str.startswith('J')]  

        df = pd.merge(feature_df, label_df, how='inner', on='iauname')
        print('Feature rows: {}, label rows: {}'.format(len(feature_df), len(label_df)))
        # assert len(df) == len(feature_df) # TODO investigate - I think features_concat includes galaxies without quality checks

        # also drop the bad ellipse rows (developed in identify_gz2_galaxies.ipynb)
        ellipse_feature_df = pd.read_parquet('/Users/walml/repos/morphology-tools/anomaly/data/decals_ellipse_features.parquet')  
        ellipse_feature_df['iauname'] = ellipse_feature_df.index.astype(str)
        bad_ellipse_features = ellipse_feature_df[np.any(ellipse_feature_df.isna(), axis=1)]
        bad_ellipse_galaxies = bad_ellipse_features['iauname']
        print('Galaxies before dropping bad ellipse features: {}'.format(len(df)))
        df = df[~df['iauname'].isin(bad_ellipse_galaxies)]
        print('Galaxies after dropping bad ellipse features: {}'.format(len(df)))

        feature_cols = [col for col in df.columns.values if col.startswith('feat')]

    else:
        raise ValueError('method {} not recognised'.format(method))

    # this will drop some rows in some configurations (e.g. filtering for featured only)
    # so only apply max_galaxies afterwards
    features, labels, responses, df = df_to_decals_training_data(
        df,
        anomalies=anomalies,
        feature_cols=feature_cols
    )

    if max_galaxies is not None:
        print('Sampling {} galaxies from {} total'.format(max_galaxies, len(labels)))
        if max_galaxies > len(labels):
            logging.warning('Not enough galaxies to sample - shuffling only')
        indices = np.arange(len(labels))
        random.shuffle(indices)
        features, labels, responses = features[indices][:max_galaxies], labels[indices][:max_galaxies], responses[indices][:max_galaxies]
        df = df.iloc[indices][:max_galaxies].reset_index()

    return features, labels, responses, df

def get_decals_label_df(anomalies, local):
    if (anomalies == 'mergers') or (anomalies == 'featured'):
        # if local:
        #     # TODO full decals predictions
        #     label_loc = '/home/walml/repos/zoobot_private/gz_decals_auto_posteriors.parquet'
        # else:
        #     label_loc = 'gz_decals_auto_posteriors.parquet'
        # switching to volunteer responses instead to not "cheat" and use ml-predicted morphology as well as ml-predicted representation
        label_loc = '/Users/walml/repos/zoobot_private/gz_decals_volunteers_5.parquet'
        label_cols = ['iauname', 'merging_merger_fraction', 'merging_total-votes', 'smooth-or-featured_total-votes', 'smooth-or-featured_featured-or-disk_fraction']  # includes responses
        if label_loc.endswith('.csv'):
            label_df = pd.read_csv(label_loc, usecols=label_cols)
        else:
            label_df = pd.read_parquet(label_loc, columns=label_cols)
    # elif anomalies == 'rings':
    #     raise NotImplementedError
    #     if local:
    #         label_loc = '/home/walml/repos/zoobot/data/ring_catalog_with_morph.csv'
    #     else:
    #         label_loc = 'ring_catalog_with_morph.csv'
    #     label_cols = None
    elif (anomalies == 'odd') or (anomalies == 'ring_responses') or (anomalies == 'irregular'):
        # if local:
        #     label_loc = '/home/walml/repos/zoobot_private/rare_features_dr5_with_ml_morph.parquet'
        # else:
        #     label_loc = 'rare_features_dr5_with_ml_morph.parquet'
        # label_cols = None
        # similarly, switch to vols and vol morphology
        rare_features_dr5 = pd.read_parquet('/Users/walml/repos/zoobot_private/rare_features_dr5.parquet')
        print('rare feature classifications: {}'.format(len(rare_features_dr5)))
        volunteers_dr5 = pd.read_parquet('/Users/walml/repos/zoobot_private/gz_decals_volunteers_5.parquet')
        print('main volunteer classifications: {}'.format(len(volunteers_dr5)))
        label_df = pd.merge(volunteers_dr5, rare_features_dr5, how='inner', on='iauname')
    else:
        raise ValueError(anomalies)

    label_df['iauname'] = label_df['iauname'].astype(str)
    # label_df['objid'] = label_df['iauname']
    print(len(label_df), 'labelled galaxies')
    return label_df


def df_to_decals_training_data(df, anomalies, feature_cols):
    print(len(df), 'galaxies with good features')

    # filter to high-ish response
    required_answer = 'smooth-or-featured'
    if anomalies == 'mergers':
        required_answer = 'merging'
    df = df[df['{}_total-votes'.format(required_answer)] >= 30]  # due to question shift, a handful have many fewer votes that total question answers would suggest
    print(len(df), 'with enough {} answers'.format(required_answer))

    if anomalies == 'mergers':
        # filter to high-ish response
        df = df[df['merging_total-votes'] >= 30]  # due to question shift, a handful have many fewer votes that total question answers would suggest
        print(len(df), 'with enough merging answers')
        responses = np.around(np.array(df['merging_merger_fraction'] * 5))  # integer responses "from" UI
        # labels = np.array(df['merging_merger_fraction'] > 0.7)  # conservative scoring following astronomaly paper
        # print('WARNING temp override merger labels')
        labels = np.array(df['merging_merger_fraction'] > 0.6)  # actually more like the same fraction of mergers as `odd' in the first benchmark, top 1.5%
    elif anomalies == 'featured':  # for debugging/slides only
        responses = np.around(np.array(df['smooth-or-featured_featured-or-disk_fraction'] * 5))  # integer responses "from" UI
        labels = np.array(df['merging_merger_fraction'] > 0.5)
    elif anomalies == 'rings':
        raise NotImplementedError
    elif anomalies == 'ring_responses':
        df = filter_to_featured_face_on(df)
        df = df.dropna(subset=['rare-features_ring_fraction'])
        print(len(df), 'with non-nan ring fractions')
        # maybe exclude some intermediate cases?
        # for frac in np.linspace(0.1, 0.7, num=100):
        #     print(frac, (df['rare-features_ring_fraction'] >= frac).mean())
        # exit()
        labels = df['rare-features_ring_fraction'].values >= 0.57  # with featured/face filter
        # labels = df['rare-features_ring_fraction'].values >= 0.46  # without featured/face filter

        responses = np.around(df['rare-features_ring_fraction'].values * 5)
    elif anomalies == 'irregular':
        df = filter_to_featured_face_on(df)
        df = df.dropna(subset=['rare-features_irregular_fraction'])
        print(len(df), 'with non-nan irregular fractions')
        # for frac in np.linspace(0.1, 0.7, num=100):
        #     print(frac, (df['rare-features_irregular_fraction'] >= frac).mean())
        # exit()
        labels = df['rare-features_irregular_fraction'].values >= 0.42  # with featured/face filter
        # labels = df['rare-features_irregular_fraction'].values >= 0.34  # without featured/face filter
        responses = np.around(df['rare-features_irregular_fraction'].values * 5)
    else:
        raise ValueError('Anomalies {} not recognised'.format(anomalies))

    features = df[feature_cols].values

    assert len(features) == len(labels)
    print(f'Features: {features.shape}')  # total num of galaxies will strongly affect results


    print('Labels: \n', pd.value_counts(labels))
    print('Responses: \n', pd.value_counts(responses))
    print('Total galaxies: {}'.format(len(labels)))

    valid_labels = np.isfinite(labels)
    if not valid_labels.all():
        raise ValueError('Bad labels: {}'.format((~valid_labels).sum()))
    valid_responses = np.isfinite(responses)
    if not valid_responses.all():
        raise ValueError('Bad responses: {}'.format((~valid_responses).sum()))

    return features, labels, responses, df


def filter_to_featured_face_on(df):
    feat = df['smooth-or-featured_featured-or-disk_fraction'] > 0.6
    face = df['disk-edge-on_no_fraction'] > 0.75
    df = df[feat & face]
    print(len(df), 'featured and face-on')
    return df


def load_gz2_data(method, anomalies, max_galaxies):

    assert anomalies == 'odd'

    if method == 'ellipse':
        feature_loc = '/Users/walml/repos/morphology-tools/anomaly/data/EllipseFitFeatures_output_back_10_12.parquet'  # michelle's version, used in the paper
        # feature_loc = 'anomaly/data/gz2_kaggle_ellipse_features.parquet'  # my version which uses the exact same example script, but is nonetheless quite different
        feature_df = pd.read_parquet(feature_loc)  # galaxy_zoo_example.py applied to full kaggle dataset
        feature_cols = feature_df.columns.values
        feature_df['objid'] = feature_df.index.astype(str)

        logging.info('All features: {}'.format(len(feature_df)))
        feature_df = feature_df.dropna(how='any')  # some features are nan
        logging.info('Non-nan features: {}'.format(len(feature_df)))
        feature_df = feature_df.reset_index(drop=True)

        label_df = pd.read_csv('/Volumes/beta/galaxy_zoo/gz2/kaggle/training_solutions_rev1.csv')  # from kaggle
        label_df['objid'] = label_df['GalaxyID'].astype(str)
        del label_df['GalaxyID']

        df = pd.merge(feature_df, label_df, how='inner', on='objid')
        assert len(feature_df) == len(df)

        # exclude galaxies with cnn features, based on precalculated venn diagram
        # see identify_gz2_galaxies.ipynb
        venn_df = pd.read_csv('/Users/walml/repos/morphology-tools/anomaly/data/gz2_galaxies_with_cnn_and_ellipse_features.csv')
        print('Galaxies before venn diagram: ', len(df))
        df = df[df['objid'].astype(str).isin(venn_df['GalaxyID'].astype(str))]
        print('Galaxies after venn diagram: ', len(df))

        df['t06_odd_a14_yes_fraction_kaggle'] = df['Class6.1']

        df.to_csv('temp_latest_forest_df.csv', index=False)

    elif method == 'cnn':
        # cnn predictions on all gz2 galaxies
        features = pd.read_parquet('/Volumes/beta/cnn_features/gz2/cnn_features_gz2.parquet')  # features and png_loc
        features['id_str'] = features['id_str'].astype(str)

        catalog = pd.read_parquet(
            '/Volumes/beta/galaxy_zoo/gz2/subjects/image_master_catalog.parquet',
            columns=['dr7objid', 't06_odd_a14_yes_fraction'])  # includes responses

        print((catalog['t06_odd_a14_yes_fraction'] > 0.9).sum())

        catalog['id_str'] = catalog['dr7objid'].astype(str)
        df = pd.merge(features, catalog, how='inner', on='id_str')

        print((df['t06_odd_a14_yes_fraction'] > 0.9).sum())

        # features and catalog should merge perfectly
        print(len(features), len(catalog), len(df))
        assert len(df) == len(features)

        feature_cols = [col for col in df.columns.values if col.startswith('feat')]
        features = df[feature_cols].values  # not yet pca'd, for now - may cache instead
        
        # #  filter to 60k subset from kaggle, not just randomly
        # kaggle_df = pd.read_csv('/Volumes/beta/galaxy_zoo/gz2/kaggle/training_solutions_rev1.csv', usecols=['GalaxyID', 'Class6.1'])  # from kaggle
        # key_df = pd.read_csv('/home/walml/Downloads/kaggle_gz_allgals_randomgalaxyid.csv', usecols=['GalaxyID', 'dr7objid'])
        # kaggle_df['GalaxyID'] = kaggle_df['GalaxyID'].astype(str)
        # key_df['GalaxyID'] = key_df['GalaxyID'].astype(str)
        # key_df['dr7objid'] = key_df['dr7objid'].astype(str)
        # kaggle_df['t06_odd_a14_yes_fraction_kaggle'] = kaggle_df['Class6.1']
        # kaggle_key_df = pd.merge(kaggle_df, key_df, on='GalaxyID', how='inner')
        # print(len(kaggle_df), len(key_df), len(kaggle_key_df))

        # print(df['id_str'])
        # print(kaggle_key_df['dr7objid'])

        # print('df before selecting kaggle only: ', len(df))
        # df['dr7objid'] = df['dr7objid'].astype(str)
        # # df = df[df['id_str'].isin(set(kaggle_key_df['dr7objid']))]
        # df = pd.merge(df, kaggle_key_df, on='dr7objid', how='inner')
        # print('df after selecting kaggle only: ', len(df))

        # precalculated version that includes dropping galaxies with nan ellipse features
        venn_df = pd.read_csv('/home/walml/repos/morphology-tools/anomaly/data/gz2_galaxies_with_cnn_and_ellipse_features.csv')
        venn_df['dr7objid'] = venn_df['dr7objid'].astype(str)
        print('Galaxies before venn diagram: ', len(df))
        # df = df[df['id_str'].astype(str).isin(venn_df['dr7objid'].astype(str))]
        df = pd.merge(df, venn_df, how='inner', left_on='id_str', right_on='dr7objid')
        print('Galaxies after venn diagram: ', len(df))
        # still need to be sure to use the kaggle labels
        df['t06_odd_a14_yes_fraction_kaggle'] = df['Class6.1']
        del df['t06_odd_a14_yes_fraction'] 

        # df.to_csv('temp_latest_cnn_df.csv', index=False)

    # if max_galaxies is not None:
    #    
    #     logging.info('Sampling {} galaxies from {} total'.format(max_galaxies, len(df)))
    #     df = df.sample(max_galaxies)

    responses = np.around(np.array(df['t06_odd_a14_yes_fraction_kaggle'] * 5))  # integer responses "from" UI
    labels = np.array(df['t06_odd_a14_yes_fraction_kaggle'] > 0.9)  # conservative scoring following astronomaly paper
    features = df[feature_cols].values

    if method == 'ellipse':
        logging.info('Applying zero mean unit variance transform to ellipse features')
        # for ellipses only, apply sklearn StandardScalar i.e. zero mean unit variance transform as per astronomaly
        scl = StandardScaler()
        features = scl.fit_transform(features)

    assert len(features) == len(labels)
    return features, labels, responses, df


def load_simulated_data():  # raw data, randomly pre-shuffled
    unshuffled_features, unshuffled_labels = load_raw_data()

    shuffle_indices= np.arange(len(unshuffled_labels))
    random.shuffle(shuffle_indices)  # inplace
    features = unshuffled_features[shuffle_indices]
    labels = unshuffled_labels[shuffle_indices].astype(int)

    # labels are originally classes?
    print('\nClasses:')
    print(pd.value_counts(labels.squeeze()))

    labels_to_responses = {
        0: 0.,
        1: 0.,
        2: 3.,
        3: 0.,
        4: 5.
    }
    # TODO use these as binary labels 
    # is_interesting = np.isclose(sorted_labels, 4)  # TODO get rws for all classes
    # is_interesting = np.isclose(sorted_labels, 5)  # use 5 if labels are responses rather than classes

    responses = labels.copy()
    for c, response in labels_to_responses.items():
        responses[labels==c] = response

    print('\nResponse labels:')
    print(pd.value_counts(responses.squeeze()))

    return features, labels, responses, None


def load_raw_data():
    labels = np.load('example_data/Simulations/labels_test.npy').reshape(-1, 1)
    features = np.load('example_data/Simulations/y_test.npy')
    assert len(features) == len(labels)
    return features, labels


def get_embed(features, n_components, save_embed='', save_variance='', new=True):

    if new:
        embedder = IncrementalPCA(n_components=n_components)
        embed = embedder.fit_transform(features) 

        if len(save_embed) > 0:
            with open(save_embed, 'wb') as f:
                pickle.dump(embed, f)
        # no train/test needed as unsupervised
        if len(save_variance) > 0:
            plt.plot(embedder.explained_variance_ratio_)  # 5 would probably do?
            logging.info('PCA with {} components preserves {}pc of variance'.format(n_components, embedder.explained_variance_ratio_.sum()))
            plt.savefig(save_variance)
            plt.close()
    else:
        raise NotImplementedError # needs care due to shuffling, except GZ2
        # assert os.path.isfile(save_embed)
        # with open(save_embed, 'rb') as f:
        #     embed = pickle.load(f)
    return embed


def get_metrics(preds, scoring_labels):
    # scoring labels: 1 where anomaly, 0 otherwise

    sort_indices = np.argsort(np.squeeze(preds))[::-1]
    # sorted_preds = preds[sort_indices]
    sorted_labels = scoring_labels[sort_indices].squeeze()

    metrics = {}
    for top_n in [50, 150]:
        metrics.update({
            'total_found_{}'.format(top_n): sorted_labels[:top_n].sum(),
            'recall_{}'.format(top_n): get_recall(sorted_labels, top_n),  # using top_n as set
            'accuracy_{}'.format(top_n): get_accuracy(sorted_labels, top_n),  # using top_n as set
            'rank_weighted_score_{}'.format(top_n): get_rank_weighted_score(sorted_labels, n=top_n)
        })
    return metrics
    

def get_recall(is_interesting, top_n):
    # fraction recovered = interesting anomalies in top n / all interesting anomalies
    return is_interesting[:top_n].sum() / is_interesting.sum()


def get_accuracy(is_interesting, top_n):
    # accuracy = % of interesting anomalies in top n
    return is_interesting[:top_n].mean()


def get_rank_weighted_score(is_interesting, n):
    # astronomaly eqn 4
    # n = "number of objects a human may reasonably look at"
    # is_interesting is the indicator: true if label in interesting class, false otherwise

    is_interesting_top = is_interesting[:n]

    i = np.arange(0, n)
    weights = n - i  # start from index=0 and lose the +1, more conventional
    # and put the -1 here in the normalisation to compensate
    s_zero = (n * (n+1) / 2)  # sum of integers from 0 to N (i.e. the weights * indicator if all are anomalies, max possible)
    return np.sum(weights * is_interesting_top) / s_zero


# def check_if_interesting(labels, anomaly_labels=[4]):
#     interesting = np.zeros_like(labels).astype(bool)  # i.e. False everywhere
#     for anomaly_label in anomaly_labels:
#         interesting = interesting | labels.astype(int) == anomaly_label
#     return interesting


def visualise_metrics(df, save_loc, total_anomalies=None):

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 12), sharex=True)

    # https://matplotlib.org/stable/users/dflt_style_changes.html#colors-in-default-property-cycle

    sns.lineplot(x='labelled_samples', y='total_found_50', data=df, label='N=50', ax=ax0, color='#1f77b4')
    ax0.axhline(50, color='#1f77b4', linestyle='--')
    sns.lineplot(x='labelled_samples', y='total_found_150', data=df, label='N=150', ax=ax0, color='#ff7f0e')
    ax0.axhline(150, color='#ff7f0e', linestyle='--')
    # if total_anomalies is not None:
    #     ax0.axhline(total_anomalies, color='r', linestyle='--')
    
    ax0.set_ylim([0., None])
    ax0.set_ylabel('Anomalies Found')

    # recall will max out if more anomalies than 50, 150, as can't possibly recover more than 50, 150
    sns.lineplot(x='labelled_samples', y='recall_50', data=df, label='N=50', ax=ax1)
    sns.lineplot(x='labelled_samples', y='recall_150', data=df, label='N=150', ax=ax1)
    if total_anomalies is not None:
            ax1.axhline(50/total_anomalies, color='#1f77b4', linestyle='--')
            ax1.axhline(150/total_anomalies, color='#ff7f0e', linestyle='--')
    # for iteration in range(df['random_state'].max()):
    #     iteration_df = df.query('random_state == {}'.format(iteration))
    #     sns.lineplot(x='labelled_samples', y='recall_50', data=iteration_df, color='k', alpha=.2, ax=ax1, legend=None)
    ax1.set_ylim([0., 1.])
    ax1.set_ylabel('Recall')

    sns.lineplot(x='labelled_samples', y='accuracy_50', data=df, label='N=50', ax=ax2)
    sns.lineplot(x='labelled_samples', y='accuracy_150', data=df, label='N=150', ax=ax2)
    # for iteration in range(df['random_state'].max()):
    #     iteration_df = df.query('random_state == {}'.format(iteration))
    #     sns.lineplot(x='labelled_samples', y='accuracy_50', data=iteration_df, color='k', alpha=.2, ax=ax2)
    ax2.set_ylim([0., 1.])
    ax2.set_ylabel('Accuracy')

    sns.lineplot(x='labelled_samples', y='rank_weighted_score_50', data=df, label='N=50', ax=ax3)
    sns.lineplot(x='labelled_samples', y='rank_weighted_score_150', data=df, label='N=150', ax=ax3)
    ax3.set_xlabel('Labelled Examples')
    ax3.set_ylabel('Rank Weighted Score')
    ax3.set_ylim([0., 1.])

    sns.lineplot(x='labelled_samples', y='score', data=df, ax=ax4)
    ax4.set_xlabel('Labelled Examples')
    ax4.set_ylabel('Supervised Score')
    ax4.set_ylim([-4, 1.])

    ax4.set_xlim([0., df['labelled_samples'].max()])

    # for ax in [ax0, ax1, ax2, ax3, ax4]:
    #     ax.set_facecolor('white')

    fig.tight_layout()
    fig.savefig(save_loc, transparent=True, facecolor='white')


def visualise_predictions_in_first_two_dims(X, preds, save_loc, xlim=None, ylim=None):
    plt.scatter(X[:, 0], X[:, 1], c=preds.squeeze())
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close()


def save_top_galaxies(preds, metadata, save_loc):
    # note that preds and metadata must align - no shuffling one without the other
    top_pred_indices = np.argsort(preds)[::-1][:12]
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12.5, 5))
    all_axes = [ax for row in axes for ax in row]
    for ax_n, ax in enumerate(all_axes):
        galaxy_n = top_pred_indices[ax_n]
        ax.imshow(Image.open(metadata.iloc[galaxy_n]['png_loc']))
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_loc, transparent=True)


def get_metrics_like_fig_5(final_sorted_labels, method, dataset_name, regression, experiment_name):
    top_n_galaxies = list(range(1, 501))  # "N" on x axis is not the num. of labels but the num. of galaxies to consider (from RWS formula)
    rank_weighted_scores = [get_rank_weighted_score(final_sorted_labels, n=top_n) for top_n in top_n_galaxies]
    fig5_metrics = {
        'top_n_galaxies': top_n_galaxies,
        'rank_weighted_scores': rank_weighted_scores,
        'method': method,
        'regression': regression
    }

    with open('anomaly/results/{}/paper_style/fig5_metrics_{}_{}.json'.format(dataset_name, regression, experiment_name), 'w') as f:
        json.dump(fig5_metrics, f)

        plt.plot(fig5_metrics['top_n_galaxies'], fig5_metrics['rank_weighted_scores'], label=regression)
        plt.xlabel('N galaxies for Rank Weighted Score')
        plt.ylabel('Rank Weighted Score')
        # plt.show()
        plt.xlim([0., 500.])
        plt.ylim([0., 1.])
        plt.tight_layout()
        plt.legend()
        plt.savefig('anomaly/results/{}/paper_style/fig5_metrics_{}_{}.png'.format(dataset_name, regression, experiment_name))
