import os
from multiprocessing import Pool

import pandas as pd
from PIL import Image
from tqdm import tqdm


def make_images(df_loc):
    df = pd.read_parquet(df_loc, columns=['iauname', 'local_png_loc'])
    pbar = tqdm(total=len(df), unit='galaxies')

    # singlethreaded version
    for _, galaxy in df.iterrows():  # about 3h
        make_single_image(galaxy)
        pbar.update()

    # # multithreaded version - maybe 30% quicker, not much difference
    # pool = Pool()  # all processors by default
    # result = pool.imap_unordered(make_single_image, df.to_dict(orient='records'))
    # [pbar.update() for _ in result]
    # pool.close()
    # pool.join()


def make_single_image(galaxy, original_size=424, crop_each_side=50, pbar=None, overwrite=False):
    save_loc = get_ring_image_path(galaxy['iauname'])  # will convert any jpeg to png
    if overwrite or not os.path.isfile(save_loc):
        im = Image.open(galaxy['local_png_loc']).crop((crop_each_side, crop_each_side, original_size-crop_each_side, original_size-crop_each_side))
        im.save(save_loc)
        if pbar:
            pbar.update()

def get_ring_image_path(iauname):
    folder = os.path.join('/Volumes/beta/galaxy_zoo/ringfinder/all', iauname[:4])
    if not os.path.exists(folder):
        os.mkdir(folder)
    return os.path.join(folder, iauname + '.png')

if __name__ == '__main__':

    # made in zoobot/notebooks/catalog_for_gz_mobile.ipynb
    make_images(df_loc='/Users/walml/repos/zoobot/notebooks/gz_mobile_catalog_preupload.parquet')
