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



def upload_galaxy(galaxy: pd.Series, project: Project):


    locations = [make_subject_images.get_ring_image_path(galaxy['iauname'])]

    subject_set_name = galaxy['subject_set']

    metadata = {
        'upload_date': str(date.today()),
        'iauname': galaxy['iauname']
    }

    ra = galaxy['ra']
    dec = galaxy['dec']

    external_links = {}

    external_links['decals_search'] = coords_to_decals_skyviewer(ra, dec)
    external_links['sdss_search'] = coords_to_sdss_navigate(ra, dec)
    external_links['panstarrs_dr1_search'] = coords_to_panstarrs(ra, dec)
    external_links['simbad_search'] = coords_to_simbad(ra, dec, search_radius=10.)
    external_links['nasa_ned_search'] = coords_to_ned(ra, dec, search_radius=10.)
    external_links['vizier_search'] = coords_to_vizier(ra, dec, search_radius=10.)

    markdown_text = {
        'decals_search': 'Click to view in DECALS',
        'sdss_search': 'Click to view in SDSS',
        'panstarrs_dr1_search': 'Click to view in PANSTARRS DR1',
        'simbad_search': 'Click to search SIMBAD',
        'nasa_ned_search': 'Click to search NASA NED',
        'vizier_search': 'Click to search VizieR'
    }

    markdown_for_external_links = {'ra': ra, 'dec': dec}
    for link_key, link_text in markdown_text.items():
        link = external_links[link_key]
        markdown_for_external_links[link_key] = wrap_url_in_new_tab_markdown(url=link, display_text=link_text)

    # print(markdown_for_external_links)
    # exit()

    metadata.update(markdown_for_external_links)
    # print(metadata)
    # exit()

    galaxy.subject_id = subject_utils.upload_subject(
        locations=locations, 
        project=project,
        subject_set_name='ring_' + subject_set_name,
        metadata=metadata)



# being lazy and pasting this here
def coords_to_simbad(ra, dec, search_radius):
    """
    Get SIMBAD search url for objects within search_radius of ra, dec coordinates.
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
        search_radius (float): search radius around ra, dec in arcseconds
    Returns:
        (str): SIMBAD database search url for objects at ra, dec
    """
    return 'http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={0}+%09{1}&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius={2}&Radius.unit=arcmin&submit=submit+query&CoordList='.format(ra, dec, search_radius)


def coords_to_decals_skyviewer(ra, dec):
    """
    Get decals_skyviewer viewpoint url for objects within search_radius of ra, dec coordinates. Default zoom.
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
    Returns:
        (str): decals_skyviewer viewpoint url for objects at ra, dec
    """
    return 'http://www.legacysurvey.org/viewer?ra={}&dec={}&zoom=15&layer=decals-dr5'.format(ra, dec)


def coords_to_sdss_navigate(ra, dec):
    """
    Get sdss navigate url for objects within search_radius of ra, dec coordinates. Default zoom.
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
    Returns:
        (str): sdss navigate url for objects at ra, dec
    """
    # skyserver.sdss.org really does skip the wwww, but needs http or link keeps the original Zooniverse root
    return 'http://skyserver.sdss.org/dr14/en/tools/chart/navi.aspx?ra={}&dec={}&scale=0.1&width=120&height=120&opt='.format(ra, dec)


def coords_to_ned(ra, dec, search_radius):
    """
    Get NASA NED search url for objects within search_radius of ra, dec coordinates.
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
        search_radius (float): search radius around ra, dec in arcseconds
    Returns:
        (str): SIMBAD database search url for objects at ra, dec
    """
    ra_string = '{:3.8f}d'.format(ra)
    dec_string = '{:3.8f}d'.format(dec)
    search_radius_arcmin = search_radius / 60.
    return 'https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0&lon={}&lat={}&radius={}&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&z_constraint=Unconstrained&z_value1=&z_value2=&z_unit=z&ot_include=ANY&nmp_op=ANY&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=Distance+to+search+center&of=pre_text&zv_breaker=30000.0&list_limit=5&img_stamp=YES'.format(ra_string, dec_string, search_radius_arcmin)


def coords_to_vizier(ra, dec, search_radius):
    """
    Get vizier search url for objects within search_radius of ra, dec coordinates.
    Include radius from search target, sort by radius from search target.
    http://vizier.u-strasbg.fr/doc/asu-summary.htx
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
        search_radius (float): search radius around ra, dec in arcseconds
    Returns:
        (str): vizier url for objects at ra, dec
    """
    return 'http://vizier.u-strasbg.fr/viz-bin/VizieR?&-c={},{}&-c.rs={}&-out.add=_r&-sort=_r'.format(
        ra, dec, search_radius)


def coords_to_panstarrs(ra, dec):
    """
    Get panstarrs dr1 cutout url at ra, dec coordinates.
    http://ps1images.stsci.edu/cgi-bin/ps1cutouts
    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
    Returns:
        (str): cutout url for objects at ra, dec
    """
    return 'http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={}{:+f}&filter=color&filter=g&filter=r&filter=i&filter=z&filter=y&filetypes=stack&auxiliary=data&size=240&output_size=0&verbose=0&autoscale=99.500000&catlist='.format(
        ra, dec)


def wrap_url_in_new_tab_markdown(url, display_text):
    return '[{}](+tab+{})'.format(display_text, url)

def main():

    subject_utils.authenticate()
    project = Project(6490)  # GZ Mobile
    df = pd.read_parquet('/Users/walml/repos/zoobot/notebooks/gz_mobile_catalog_preupload.parquet')
    catalog = pd.read_parquet('/Users/walml/repos/zoobot_private/gz_decals_auto_posteriors.parquet', columns=['ra', 'dec', 'iauname'])  # for ra/dec
    df = pd.merge(df, catalog, on='iauname', how='inner')

    print(df['subject_set'].value_counts())
    # exit()

    subject_set_to_upload = 'prioritised_remaining'

    df_to_upload = df.query('subject_set== "{}"'.format(subject_set_to_upload))


    df_to_upload['subject_set'] = ['prioritised_remaining_a'] * 5000 + ['prioritised_remaining_b'] * 5000 + ['prioritised_remaining_c'] * 5000 + ['prioritised_remaining_d'] * 5000 + ['prioritised_remaining_e'] * 5000 + ['prioritised_remaining_f'] * 187
    print(df_to_upload['subject_set'].value_counts())
    # exit()

    for subject_set in df_to_upload['subject_set'].unique():
        print(subject_set)
        df_those_subjects = df_to_upload.query('subject_set== "{}"'.format(subject_set))
        for _, galaxy in tqdm.tqdm(df_those_subjects.iterrows(), total=len(df_those_subjects), unit='galaxies'):
            upload_galaxy(galaxy, project)

if __name__ == '__main__':
    main()
