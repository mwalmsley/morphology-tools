# TODO copy from CHIME interfaces ubject uplaoad code, no db

import json
import os
from datetime import date
import logging
from typing import List, Dict

import tqdm
import numpy as np
import pandas as pd
from panoptes_client import Panoptes, Project, SubjectSet, Subject

from shared_astro_utils import subject_utils
import make_subject_images



def upload_galaxy(galaxy, project):


    locations = [make_subject_images.get_ring_image_path(galaxy['iauname'])]

    subject_set_name = galaxy['subject_set']

    metadata = {
        'upload_date': str(date.today()),
        'iauname': galaxy['iauname']
    }

    galaxy.subject_id = subject_utils.upload_subject(
        locations=locations, 
        project=project,
        subject_set_name='ring_' + subject_set_name,
        metadata=metadata)


def main():

    subject_utils.authenticate()
    project = Project(6490)  # GZ Mobile
    df = pd.read_parquet('/Users/walml/repos/zoobot/notebooks/gz_mobile_catalog_preupload.parquet').sample(1000)

    subject_set_to_upload = 'prioritised_launch_1k'

    df_to_upload = df.query('subject_set== "{}"'.format(subject_set_to_upload))

    for _, galaxy in tqdm.tqdm(df_to_upload.iterrows(), total=len(df_to_upload), unit='galaxies'):
        upload_galaxy(galaxy, project)

if __name__ == '__main__':
    main()