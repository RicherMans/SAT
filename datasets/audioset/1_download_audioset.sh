# Downloading audioset script
# Requirements are proxychains and ffmpeg
# Also Youtube-dl
# pip install youtube-dl


# Chan be changed by ./0_download_audioset.sh 8
base_dir=${1-"./data"}
njobs=${2:-64}
SAMPLE_RATE=${3:-16000}

EXTENSION="wav"

balanced_dir=${base_dir}/audio/balanced/
eval_dir=${base_dir}/audio/eval/
csv_dir=${base_dir}/csvs/
meta_dir=${base_dir}/metadata/
log_dir=${base_dir}/logs/
psl_dir=${base_dir}/psl_labels/

fetch_clip() {
    # echo "Fetching $1 ($2 to $3)..."
    outname="$1_$2_$3"
    outdir=${4}
    output_path=${outdir}/${outname}.${EXTENSION}

    # Do not redownload already existing file
    if [ -f "${output_path}" ]; then
        return
    fi
    link=$(yt-dlp --force-ipv4 -g https://youtube.com/watch?v=$1 | awk 'NR==2{print}')

    if [ $? -eq 0 ]; then
        ffmpeg -loglevel quiet  -i "$link" -ar $SAMPLE_RATE -ac 1 -ss "$2" -to "$3" "${output_path}"
    fi
}


function parallel_download() {
    if [[ $# != 2 ]]; then
        echo "[csv_segments] [output_dir]"
        exit
    fi
    csv_segments=${1}
    output_dir=${2}
    echo "Downloading ${csv_segments} Subset using ${njobs} workers"
    grep "^[^#;]" ${csv_segments} | parallel --resume --joblog ${log_dir}/job.log -j $njobs --colsep=, fetch_clip {1} {2} {3} ${output_dir} > /dev/null
}


export SAMPLE_RATE
export EXTENSION
export -f fetch_clip

mkdir -p ${csv_dir} ${balanced_dir} ${eval_dir} ${log_dir} ${meta_dir} ${psl_dir}

function download_zenodo(){
    output_file=$(echo $1 | awk -F['?/'] '{print $(NF-1)}')
    wget --continue $1 -O "${2}/${output_file}"
}

# PSL Data in Parquet format
download_zenodo "https://zenodo.org/record/7964975/files/full_vit_base_chunked2s_topk10_47_40.parquet?download=1" "${psl_dir}"
download_zenodo "https://zenodo.org/record/7964975/files/full_vit_base_chunked4s_topk10_47_40.parquet?download=1" "${psl_dir}"
download_zenodo "https://zenodo.org/record/7964975/files/balanced_vit_base_chunked4s_topk10_47_40.parquet?download=1" "${psl_dir}"
download_zenodo "https://zenodo.org/record/7964975/files/balanced_vit_base_chunked2s_topk10_47_40.parquet?download=1" "${psl_dir}"

wget --continue "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" -O "${meta_dir}/class_labels_indices.csv"

wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv -O ${csv_dir}/balanced_train_segments.csv
wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv  -O ${csv_dir}/eval_segments.csv
wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv -O ${csv_dir}/class_labels_indices.csv


parallel_download ${csv_dir}/balanced_train_segments.csv ${balanced_dir}

parallel_download ${csv_dir}/eval_segments.csv ${eval_dir}

echo "Finished Downloading data"

