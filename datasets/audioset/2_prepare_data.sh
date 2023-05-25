#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DUMP_WAVE_TO_HDF_SCRIPT=${SCRIPT_DIR}"/../utils/dump_audio_to_hdf5.py"
PREPARE_LABEL_SCRIPT=${SCRIPT_DIR}"/../utils/prepare_label_list.py"

## Prepare data, use defaults from download_audioset
base_dir=${1-"./data"}
output_label_dir=${2-$PWD"/data/labels"}
output_hdf5_dir=${3-$PWD"/data/hdf5"}

balanced_dir=${base_dir}/audio/balanced/
eval_dir=${base_dir}/audio/eval/
csv_dir=${base_dir}/csvs/


mkdir -p ${output_hdf5_dir} ${output_label_dir}

function dump_wave_to_hdf5() {
    input_csv=$1
    outputhdf=$2
    output_final_labels=$3
    echo "Dumping data from $input_csv raw wave to hdf5 in ${outputhdf}"
    python3 ${DUMP_WAVE_TO_HDF_SCRIPT} $input_csv -o ${outputhdf}

    name=${input_csv##*/}
    head -n 1 ${input_csv} | awk '{print $0"\thdf5path"}' > ${output_final_labels}
    cat $input_csv | awk -v f=${outputhdf} 'NR>1{print $0"\t"f }' >> ${output_final_labels}
}

# BALANCED DATA

balanced_temp_file=$(mktemp -t balancedXXXXX.csv)
echo "Processing balanced subset"

python3 ${PREPARE_LABEL_SCRIPT} ${balanced_dir} ${csv_dir}/balanced_train_segments.csv ${csv_dir}/class_labels_indices.csv $balanced_temp_file
dump_wave_to_hdf5 ${balanced_temp_file} ${output_hdf5_dir}/balanced.h5 ${output_label_dir}/balanced.csv

#EVAL DATA

echo "Processing eval subset"
eval_temp_file=$(mktemp -t evalXXXXX.csv)
python3 ${PREPARE_LABEL_SCRIPT} ${eval_dir} ${csv_dir}/eval_segments.csv ${csv_dir}/class_labels_indices.csv ${eval_temp_file}

dump_wave_to_hdf5 ${eval_temp_file} ${output_hdf5_dir}/eval.h5 ${output_label_dir}/eval.csv


echo "Preparation data done, files are at ${output_label_dir}/balanced.csv and ${output_label_dir}/eval.csv"
