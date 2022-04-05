# Script to recreate plots in the paper
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
from astronomaly.utils.utils import get_visualisation_sample
import os
import pandas as pd
import numpy as np

"""
Root directory for data
data_dir = os.path.join(os.path.sep, 'home', 'michelle', 'BigData', 
                        'Anomaly', '')
"""
data_dir = os.path.join('/home/walml/repos/morphology-tools/anomaly')

# Where the galaxy zoo images are
"""
image_dir = os.path.join(data_dir, 'GalaxyZoo', 
                         'galaxy-zoo-the-galaxy-challenge', 
                         'images_training_rev1', '')
"""
image_dir = os.path.join(data_dir, 'data', 'images_training_rev1')

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output_replication', 'images', 'galaxy_zoo', '')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
image_transform_function = [
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_scale]

# You can apply a different set of transforms to the images that get displayed
# in the frontend. In this case, I want to see the original images before sigma
# clipping is applied.
display_transform_function = [
    image_preprocessing.image_transform_scale]

# For galaxy zoo, we're lucky enough to have some actual human labelled data
# that we use to illustrate the human-in-the-loop learning instead of having 
# to label manually
"""
df = pd.read_csv(os.path.join(data_dir, 'GalaxyZoo', 
                            'galaxy-zoo-the-galaxy-challenge', 
                            'training_solutions_rev1.csv'),
                index_col=0)
"""
df = pd.read_csv(
    '/media/walml/beta1/galaxy_zoo/gz2/kaggle/training_solutions_rev1.csv',
    index_col=0)
df.index = df.index.astype(str)
additional_metadata = df[['Class6.1']]


def run_pipeline():
    """
    Any script passed to the Astronomaly server must implement this function.
    run_pipeline must return a dictionary that contains the keys listed below.

    Parameters
    ----------

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data. Keys must include: 
        'dataset' - an astronomaly Dataset object
        'features' - pd.DataFrame containing the features
        'anomaly_scores' - pd.DataFrame with a column 'score' with the anomaly
        scores
        'visualisation' - pd.DataFrame with two columns for visualisation
        (e.g. TSNE or UMAP)
        'active_learning' - an object that inherits from BasePipeline and will
        run the human-in-the-loop learning when requested

    """
    # The galaxy zoo data takes long to run just because it takes time to read
    # each file in from the file system. Uncomment this to use less data.
    # fls = os.listdir(image_dir)[:1000]

    # This creates the object that manages the data
    image_dataset = image_reader.ImageThumbnailsDataset(
        directory=image_dir, output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        additional_metadata=additional_metadata,
        # list_of_files=fls
    )

    # Creates a pipeline object for feature extraction
    pipeline_ellipse = shape_features.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0],
        output_dir=output_dir, channel=0, force_rerun=True, 
        central_contour=False)

    # Actually runs the feature extraction
    
    """
    ## features = pipeline_ellipse.run_on_dataset(image_dataset)
    # features = pd.read_parquet(output_dir+'EllipseFitFeatures_output.parquet')
    """
    features = pd.read_parquet('anomaly/data/EllipseFitFeatures_output_back_10_12.parquet')

    # Now we rescale the features using the same procedure of first creating
    # the pipeline object, then running it on the feature set
    pipeline_scaler = scaling.FeatureScaler(force_rerun=True,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it
    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=True, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=True, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)

    # This is unique to galaxy zoo which has labelled data. We first sort the
    # data by anomaly score and then "label" the N most anomalous objects
    anomalies = anomalies.sort_values('score', ascending=False)
    N_labels = 200
    # The keyword 'human_label' must be used to tell astronomaly which column
    # to use for the HITL
    anomalies['human_label'] = [-1] * len(anomalies)
    inds = anomalies.index[:N_labels]
    human_probs = additional_metadata.loc[inds, 'Class6.1']
    human_scores = np.round(human_probs * 5).astype('int')
    anomalies.loc[inds, 'human_label'] = human_scores.astype(int)

    try:
        # This is used by the frontend to store labels as they are applied so
        # that labels are not forgotten between sessions of using Astronomaly
        if 'human_label' not in anomalies.columns:
            df = pd.read_csv(
                os.path.join(output_dir, 'ml_scores.csv'), 
                index_col=0,
                dtype={'human_label': 'int'})
            df.index = df.index.astype('str')

            if len(anomalies) == len(df):
                anomalies = pd.concat(
                    (anomalies, df['human_label']), axis=1, join='inner')
    except FileNotFoundError:
        pass

    # This is the active learning object that will be run on demand by the
    # frontend 
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir, force_rerun=True)

    # We use TSNE for visualisation which is run in the same way as other parts
    # of the pipeline.
    # I give it a few anomalies and then a random sample just so the plot is 
    features_to_plot = get_visualisation_sample(features, anomalies, 
                                                anomaly_column='score',
                                                N_anomalies=20,
                                                N_total=2000)
    """
    # pipeline_tsne = tsne.TSNE_Plot(
    #     force_rerun=True,
    #     output_dir=output_dir,
    #     perplexity=100)
    # t_plot = pipeline_tsne.run(features_to_plot)
    """
    t_plot = None  # no need here

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}


results = run_pipeline()

# my stuff below

anomaly_scores = results['anomaly_scores']  # sorted by most to least anomalous
features = results['features']  # not sorted

print(anomaly_scores.head())
print(features.head())

features_and_labels = pd.merge(features, anomaly_scores, how='inner', left_index=True, right_index=True)  # merge on indices (objid/galaxyID)
print(features_and_labels.head())

trained_score_df = results['active_learning']._execute_function(features_and_labels)
print(trained_score_df.head())

result_df = pd.merge(trained_score_df, anomaly_scores, how='inner', left_index=True, right_index=True)
additional_metadata['is_anomaly'] = additional_metadata['Class6.1'] > 0.9  # > not >= removes 3 anomlies (924 vs 927)
print(additional_metadata['is_anomaly'].value_counts())
result_df = pd.merge(result_df, additional_metadata[['is_anomaly']], how='inner', left_index=True, right_index=True)

name = 'original_65579fb_repeat4_gp9'  
result_df.to_csv('{}.csv'.format(name))

final_sorted_labels = result_df['is_anomaly'][np.argsort(result_df['trained_score'])[::-1]]

import shared
shared.get_metrics_like_fig_5(final_sorted_labels, method='original', dataset_name='gz2', regression='forest', experiment_name=name)

# matches the paper perfectly at commit 65579fb (oct 17 2020) and at d1b57d82 (latest main, basically)
