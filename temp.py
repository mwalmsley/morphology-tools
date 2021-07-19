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


    # mine_active = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_mine_sorted.json', 'r'))
    # mine_noactive = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_mine_noactive.json', 'r'))
    # loch_active = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_lochner_sorted.json', 'r'))
    # loch_noactive = json.load(open('anomaly/results/gz2/paper_style/fig5_metrics_baseline_lochner_noactive.json', 'r'))

    # plt.plot(mine_active['top_n_galaxies'], mine_active['rank_weighted_scores'], color='b', label='Mine Active')
    # plt.plot(mine_noactive['top_n_galaxies'], mine_noactive['rank_weighted_scores'], color='orange', label='Mine Not')

    # plt.plot(loch_active['top_n_galaxies'], loch_active['rank_weighted_scores'], color='b', label='Loch Active', linestyle='--')
    # plt.plot(loch_noactive['top_n_galaxies'], loch_noactive['rank_weighted_scores'], color='orange', label='Loch Not', linestyle='--')

    # # for (experiment, label, color) in [(cnn, 'GP+CNN', 'xkcd:mid blue'), (ellipse, 'GP+Ellip. (new)', 'xkcd:soft green'), (ellipse_loch, 'GP+Ellip. (orig.)', 'xkcd:purple pink')]:
    # #     for run in experiment:
    # #         plt.plot(run['top_n_galaxies'], run['rank_weighted_scores'], alpha=.2, color=color)
    # #     mean_scores = np.mean(np.array([run['rank_weighted_scores'] for run in experiment]), axis=0)
    # #     plt.plot(run['top_n_galaxies'], mean_scores, color=color, label=label)
    # plt.legend()
    # plt.xlim([0., 500.])
    # plt.ylim([0., 1.])
    # plt.xlabel('N galaxies for Rank Weighted Score')
    # plt.ylabel('Rank Weighted Score')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('fig5_metrics_active_feature_comparison_b.png')

    original = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_original_*gp9.json')]  # 655, oct 2020 commit
    updated = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_replication_updated_416*gp9.json')]  # 416, my first gz2 commit
    gz2 = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_replication_updated_416*gp9.json')]
    # scratch = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_lochner_active_old_distances_repeat*.json')]
    scratch = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_lochner_active_old_distances_repeat*scaled*.json')]
    
    # main current issue? Doesn't work very well
    gp_ellipse = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_latest_ellipse_loch_repeat*.json')]
    # gp_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_latest_cnn_repeat*.json')]
    # if_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_IF_repeat*.json')]
    # random_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_random_repeat*.json')]
    random_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_random_staticdata_random10_*.json')]
    # gp_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_random_maxunc_repeat*.json')]
    # forest_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_forest_latest_cnn_repeat*.json')]
    twenty_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_20comp_*.json')]
    forty_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_40comp_*.json')]
    forty_cnn_rep = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_replication_comp40_*.json')]
    # maxei_cnn = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_retrain1_maxei40_comp40*.json')]
    forty_cnn_human = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_knownlabels_comp40_*.json')]
    forty_cnn_human_rep = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_knownlabels_replication_comp40_*.json')]

    irregular = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/decals/paper_style/fig5_metrics_gp_cnn_irregular*.json')]
    mergers = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/decals/paper_style/fig5_metrics_gp_cnn_mergers*.json')]
    ring_responses = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/decals/paper_style/fig5_metrics_gp_cnn_ring_responses*.json')]

    odd_explore_p5 = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_odd_comp40_trade0p5_debug_*.json')]
    assert odd_explore_p5

    odd_explore_retrain4 = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_odd_comp40_trade0p5_retrain3_*.json')]
    assert odd_explore_retrain4

    odd_explore_retrain1 = [json.load(open(loc, 'r')) for loc in glob.glob('/home/walml/repos/morphology-tools/anomaly/results/gz2/paper_style/fig5_metrics_gp_cnn_odd_comp40_trade0p5_retrain1_*.json')]
    assert odd_explore_retrain1

    assert original
    assert updated
    assert gz2
    assert scratch
    assert gp_ellipse
    assert forty_cnn_human

    print(len(original))
    print(len(updated))
    print(len(gz2))
    print(len(scratch))
    print(len(forty_cnn_rep) + len(forty_cnn))

    # [(original, 'oct-20 main', 'xkcd:mid blue'), (updated, 'latest main', 'xkcd:soft green'), (gz2, 'latest gz2', 'xkcd:purple pink'), (scratch, 'Scratch', 'black')]:
    # [(forty_cnn, '40comp', 'xkcd:mid blue'), (random_cnn, '10comp', 'xkcd:purple pink'), (scratch, 'scratch', 'xkcd:soft green'), (forty_cnn_rep, '40compv2', 'black')]:
    # for (experiment, label, color) in [(forty_cnn_human + forty_cnn_human_rep, 'gp+cnn+h', 'xkcd:purple pink'), (scratch, 'baseline', 'black')]:
    # for (experiment, label, color) in [(irregular, 'decals-irregular', 'xkcd:purple pink'), (mergers, 'decals-merger', 'xkcd:mid blue'), (ring_responses, 'decals-rings', 'black')]:
    # for (experiment, label, color) in [(odd_explore_p5, 'p5', 'xkcd:mid blue'), (odd_explore_retrain4, 'retrain4', 'xkcd:soft green'), (forty_cnn, '40comp', 'black')]:
    for (experiment, label, color) in [(odd_explore_p5, 'batch=10', 'xkcd:mid blue'), (odd_explore_retrain4, 'batch=4', 'xkcd:soft green'), (odd_explore_retrain1, 'batch=1', 'black')]:
        for run in experiment:
            plt.plot(run['top_n_galaxies'], run['rank_weighted_scores'], alpha=.2, color=color)
        mean_scores = np.mean(np.array([run['rank_weighted_scores'] for run in experiment]), axis=0)
        linestyle=None
        # if label == 'Scratch':
        #     linestyle = '--'
        plt.plot(run['top_n_galaxies'], mean_scores, color=color, label=label, linestyle=linestyle)
    plt.legend()
    plt.xlim([0., 500.])
    plt.ylim([0., 1.])
    plt.xlabel('N galaxies for Rank Weighted Score')
    plt.ylabel('Rank Weighted Score')
    plt.tight_layout()
    plt.show()
    # plt.savefig('fig5_metrics_comparison_latest_two.png')
    # plt.savefig('fig5_metrics_comparison_decals.png')
    # plt.savefig('fig5_metrics_comparison_latest_two.png')
