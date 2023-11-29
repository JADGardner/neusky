import os
import shutil
import requests
from zipfile import ZipFile
import argparse

def download_and_unzip(url, download_dir):
    # Extract the base file name without query parameters
    local_zip_file_name = url.split('/')[-1].split('?')[0]
    local_zip_file = os.path.join(download_dir, local_zip_file_name)

    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download the file from the URL
    response = requests.get(url)
    with open(local_zip_file, 'wb') as file:
        file.write(response.content)

    # Unzip the file
    with ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    # Remove the zip file
    os.remove(local_zip_file)

def copy_segmentation_masks(source_base_dir, target_base_dir, splits, scene):
    for split in splits:
        source_dir = os.path.join(source_base_dir, scene, split, 'cityscapes_mask')
        target_dir = os.path.join(target_base_dir, 'Data', scene, 'final', split, 'cityscapes_mask')

        if not os.path.exists(source_dir):
            print(f"Source directory not found: {source_dir}")
            continue

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)

            if os.path.isdir(source_item):
                if os.path.exists(target_item):
                    shutil.rmtree(target_item)
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)


def main():
    parser = argparse.ArgumentParser(description='Process scene data.')
    parser.add_argument('scene', help='Scene name')
    parser.add_argument('target_directory', help='Target directory, /path/to/NeRF-OSR')

    args = parser.parse_args()
    scene = args.scene
    target_base_directory = args.target_directory

    scenes = {
      'europa': 'https://www.dropbox.com/scl/fi/swm3oy0667vivg3mnzucu/europa.zip?rlkey=4g043vqbz256il37t3z3f0xec&dl=1',
      'lk2': 'https://www.dropbox.com/scl/fi/9ebexftvbgftar68hgmf4/lk2.zip?rlkey=1yxgvpxcqrlvvda2qsvwsta6d&dl=1',
      'lwp': 'https://www.dropbox.com/scl/fi/ioy22gpisa8e2p8vph847/lwp.zip?rlkey=n2av9bnbmixpflux5gotfd27h&dl=1',
      'rathaus': 'https://www.dropbox.com/scl/fi/9erqpaw845ycjdwhwqjh7/rathaus.zip?rlkey=pwdgbp5yfegpfs6lcrkj1tw04&dl=1',
      'schloss': 'https://www.dropbox.com/scl/fi/heobhkelfqn4qgfeg89qn/schloss.zip?rlkey=l770hx8tjmidlvjgj3n0vcs4x&dl=1',
      'st': 'https://www.dropbox.com/scl/fi/dcdhfi4zdbl8vakq81qo9/st.zip?rlkey=sdd6k8qedakfhzi8kpfikzrr7&dl=1',
      'stjacob': 'https://www.dropbox.com/scl/fi/2m56j82x7196bhu6egbul/stjacob.zip?rlkey=q88oywyadchgw2p3zib5hgqjb&dl=1'
    }

    if scene not in scenes:
        print(f"Scene {scene} not on of: europa, lk2, lwp, rathaus, schloss, st, stjacob.")
        return

    url = scenes[scene]
    splits = ['test', 'train', 'validation']
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the temp directory relative to the script directory
    source_base_directory = os.path.join(script_dir, 'temp', 'cityscapes_masks_for_scenes')

    download_and_unzip(url, source_base_directory)
    copy_segmentation_masks(source_base_directory, target_base_directory, splits, scene)

    # Delete the source directory after copying is done
    shutil.rmtree(os.path.join(script_dir, 'temp'))

if __name__ == "__main__":
    main()