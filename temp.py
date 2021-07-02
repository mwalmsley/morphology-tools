import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')



if __name__ == '__main__':
    
    # df_lochner = pd.read_parquet('anomaly/data/EllipseFitFeatures_output_back_10_12.parquet')
    # print(df_lochner.head())
    # print(len(df_lochner))

    # df_mike = pd.read_parquet('anomaly/data/gz2_kaggle_ellipse_features.parquet')
    # print(df_mike.head())
    # print(len(df_mike))

    # # they are not the same!

    # with open('fig5_metrics_baseline_distances0p01.json', 'r') as f:
    #     fig5_baseline = json.load(f)

    # print(fig5_baseline)
    # plt.plot(fig5_baseline['top_n_galaxies'], fig5_baseline['rank_weighted_scores'])
    # plt.xlabel('N galaxies for Rank Weighted Score')
    # plt.ylabel('Rank Weighted Score')
    # # plt.show()
    # plt.xlim([0., 500.])
    # plt.tight_layout()
    # plt.savefig('fig5_metrics_baseline_distances0p01.png')

    # cnn = [json.load(open(loc, 'r')) for loc in glob.glob('fig5_metrics_gp_default*.json')]
    # ellipse = [json.load(open(loc, 'r')) for loc in glob.glob('fig5_metrics_gp_ellipse*.json') if 'loch' not in loc]
    # ellipse_loch = [json.load(open(loc, 'r')) for loc in glob.glob('fig5_metrics_gp_ellipse_loch*.json')]

    # for (experiment, label, color) in [(cnn, 'GP+CNN', 'xkcd:mid blue'), (ellipse, 'GP+Ellip. (new)', 'xkcd:soft green'), (ellipse_loch, 'GP+Ellip. (orig.)', 'xkcd:purple pink')]:
    #     for run in experiment:
    #         plt.plot(run['top_n_galaxies'], run['rank_weighted_scores'], alpha=.2, color=color)
    #     mean_scores = np.mean(np.array([run['rank_weighted_scores'] for run in experiment]), axis=0)
    #     plt.plot(run['top_n_galaxies'], mean_scores, color=color, label=label)
    # plt.legend()
    # plt.xlim([0., 500.])
    # plt.ylim([0., 1.])
    # plt.xlabel('N galaxies for Rank Weighted Score')
    # plt.ylabel('Rank Weighted Score')
    # plt.tight_layout()
    # plt.savefig('fig5_metrics_gp_comparison.png')


    mine_active = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_mine_sorted.json', 'r'))
    mine_noactive = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_mine_noactive.json', 'r'))
    loch_active = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_lochner_sorted.json', 'r'))
    loch_noactive = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_lochner_noactive.json', 'r'))

    plt.plot(mine_active['top_n_galaxies'], mine_active['rank_weighted_scores'], color='b', label='Mine Active')
    plt.plot(mine_noactive['top_n_galaxies'], mine_noactive['rank_weighted_scores'], color='orange', label='Mine Not')

    plt.plot(loch_active['top_n_galaxies'], loch_active['rank_weighted_scores'], color='b', label='Loch Active', linestyle='--')
    plt.plot(loch_noactive['top_n_galaxies'], loch_noactive['rank_weighted_scores'], color='orange', label='Loch Not', linestyle='--')

    # for (experiment, label, color) in [(cnn, 'GP+CNN', 'xkcd:mid blue'), (ellipse, 'GP+Ellip. (new)', 'xkcd:soft green'), (ellipse_loch, 'GP+Ellip. (orig.)', 'xkcd:purple pink')]:
    #     for run in experiment:
    #         plt.plot(run['top_n_galaxies'], run['rank_weighted_scores'], alpha=.2, color=color)
    #     mean_scores = np.mean(np.array([run['rank_weighted_scores'] for run in experiment]), axis=0)
    #     plt.plot(run['top_n_galaxies'], mean_scores, color=color, label=label)
    plt.legend()
    plt.xlim([0., 500.])
    plt.ylim([0., 1.])
    plt.xlabel('N galaxies for Rank Weighted Score')
    plt.ylabel('Rank Weighted Score')
    plt.tight_layout()
    # plt.show()
    plt.savefig('fig5_metrics_active_feature_comparison_b.png')
