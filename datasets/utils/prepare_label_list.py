import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import audioread

parser = argparse.ArgumentParser()
parser.add_argument('root_data_dir', help='Root of downloaded directory')
parser.add_argument(
    'segments_csv',
    help=
    'Downloaded segments.csv file eg., http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
)
parser.add_argument(
    'class_labels_indices',
    help='Class label indicies file, mapping mid to an integer')
parser.add_argument('output_csv', help='Parsed output')
args = parser.parse_args()

df = pd.read_csv(
    args.segments_csv,
    sep='\s+',
    skiprows=3,
    engine='python',
    names=['file_id', 'start', 'end', 'labels'],
    dtype=object)  # Pass dtype=object in order to not cast start/end to floats
# Remove , at the end for each item
df = df[df.columns].replace(',$', '', regex=True)
class_maps_df = pd.read_csv(args.class_labels_indices, sep=',')
mid_to_index = class_maps_df.set_index('mid')['index'].to_dict()

absolute_root = Path(args.root_data_dir).absolute()


def get_duration(wavefile):
    with audioread.audio_open(wavefile) as rp:
        return round(rp.duration, 1)

def return_fname_and_filter(row):
    fname = absolute_root / f"{row['file_id']}_{row['start']}_{row['end']}.wav"
    # print(fname)
    if fname.exists() and fname.stat().st_size != 0:  #Not empty
        idx_labels = [
            str(mid_to_index[x]) for x in row['labels'].strip("\"").split(',')
        ]
        readable_labels = ";".join(idx_labels)
        try:
            duration = get_duration(fname)
        except EOFError:
            # Return nothing if the file has some problems
            return pd.Series({
                'filename': None,
                'labels': None,
                'duration': None
            })
        if duration > 0.1:
            return pd.Series({
                'filename': fname,
                'labels': readable_labels,
                'duration': duration
            })
    return pd.Series({'filename': None, 'labels': None, 'duration': None})


tqdm.pandas()
df = df.progress_apply(return_fname_and_filter,
                       axis=1).dropna().reset_index(drop=True)
df.to_csv(args.output_csv, sep='\t', index=False)
