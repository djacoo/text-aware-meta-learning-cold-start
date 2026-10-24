import os
import io
import zipfile
import urllib.request
import datetime
import pandas as pd

def download_dataset(target_dir="data/ml-1m"):
    """
    Check if ml-1m dataset already exists (including extra files from MeLU) in target_dir.
    If not, download it from the original repo and return the path
    """

    if os.path.exists(target_dir):
        print(f"Dataset already exists int {target_dir}.\n")
        return target_dir

    print(f"Dataset not found in {target_dir}. Download started...\n")
    os.makedirs(target_dir, exist_ok=True)

    url = "https://github.com/hoyeoplee/MeLU/archive/refs/heads/master.zip"
    req = urllib.request.urlopen(url)

    with zipfile.ZipFile(io.BytesIO(req.read())) as z:
        prefix = "MeLU-master/movielens/ml-1m/"
        for member in z.namelist():
            if member.startswith(prefix) and not member.endswith("/"):
                filename = os.path.basename(member)
                target_file_path = os.path.join(target_dir, filename)
                with z.open(member) as source, open(target_file_path, "wb") as target:
                    target.write(source.read())

    print(f"Download and extraction succesfully completed in {target_dir}.\n")
    return target_dir

class movielens_1m(object):
    def __init__(self, dataset_path="data/ml-1m"):
        self.user_data, self.item_data, self.score_data = self.load(dataset_path)

    def load(self, dataset_path):
        path = dataset_path
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'], 
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], 
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data