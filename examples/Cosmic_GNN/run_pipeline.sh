#! /bin/env bash
REDO=false
base_path=/mnt/data1/tkar/cosmic_new
sample_name=cosmic_michel_f3
sample_identifier=aa

if [[ "$REDO" == true ]]; then
    echo ${base_path}/feature_store/${sample_name}/${sample_identifier}  
    echo ${base_path}/fully_connected_graphs/${sample_name}/${sample_identifier}
    echo ${base_path}/gnn/${sample_name}/${sample_identifier}/cm_m_mixed_gnn
    echo ${base_path}/results/edge_level/${sample_identifier}/cm_m_gnn_signal
    rm -r ${base_path}/feature_store/${sample_name}/${sample_identifier}  
    rm -r ${base_path}/fully_connected_graphs/${sample_name}/${sample_identifier}
    rm -r ${base_path}/gnn/${sample_name}/${sample_identifier}/cm_m_mixed_gnn
    rm -r ${base_path}/results/edge_level/${sample_identifier}/cm_m_gnn_signal
fi

# ============ STAGE: Data Reading ============
# ***produce csv files from the raw data***

cd yaml_files

yaml_file=cosmic_reader_v2.yaml
input_dir=${base_path}/raw_data/split_${sample_name}/
input_file=${input_dir}/${sample_name}${sample_identifier}.csv
stage_dir=${base_path}/feature_store/${sample_name}/${sample_identifier}/
data_split_file=dataset_size_cm_f3${sample_identifier}.csv

echo Yaml File for the Data Reading Stage: ${yaml_file}
echo Input Directory for the Data Reading Stage: ${input_dir}
echo Stage Directory for the Data Reading Stage: ${stage_dir}

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@input_file.*@input_file: ${input_file}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split_file.*@data_split_file: ${data_split_file}@g" ${yaml_file}

#acorn infer ${yaml_file}

# ***procude the feature store with truth graphs***
yaml_file=graph_constr.yaml
input_dir=${stage_dir}
stage_dir=${base_path}/fully_connected_graphs/${sample_name}/${sample_identifier}/

n_trainset=$(find ${input_dir}/trainset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m)
n_valset=$(find ${input_dir}/valset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m)  
n_testset=$(find ${input_dir}/testset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m) 
data_split=\[${n_trainset},${n_valset},${n_testset}\]
echo Data Split: ${data_split}

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file} 

echo Yaml File for the Graph Construction Stage: ${yaml_file}
echo Input Directory for the Data Reading Stage: ${input_dir}
echo Stage Directory for the Data Reading Stage: ${stage_dir}

#acorn infer ${yaml_file}
#================================================================================================

# *** remove weird events ***
cd ..
echo $'\nRemoving weird events\n'
input_dir=${stage_dir}
echo Input Directory for the weird events remover in the python script: ${input_dir}$'\n'
sed -i -e "s@input_dir =.*@input_dir = '${input_dir}'@g" remove_weird_events.py
#python remove_weird_events.py

cd yaml_files

# ============== STAGE: Edge Classification ==============
# *** infer the edge classification using a model specified by the checkpoint ***
echo $'\nEdge Classification\n'
check_point='/mnt/data1/karres/cosmics_test/cm_m_mixed_gnn/artifacts/best-b575lgmu-val_loss=0.000902-epoch=90.ckpt'
stage_dir=${base_path}/gnn/${sample_name}/${sample_identifier}/cm_m_mixed_gnn
devices=\[7\]
yaml_file=gnn_infer.yaml

n_trainset=$(find ${input_dir}/trainset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m)
n_valset=$(find ${input_dir}/valset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m)  
n_testset=$(find ${input_dir}/testset -maxdepth 1 -name "*.pyg" -printf '.' | wc -m) 
data_split=\[${n_trainset},${n_valset},${n_testset}\]

echo Data Split: ${data_split}   

echo Yaml File for the Edge Classification Stage: ${yaml_file}
echo Input Directory for the Edge Classification Stage: ${input_dir}
echo Stage Directory for the Edge Classification Stage: ${stage_dir}

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file}
sed -i -e "s@devices.*@devices: ${devices}@g" ${yaml_file}

echo $'Running inference for the edge classification\n'
#acorn infer ${yaml_file} -c ${check_point}

# *** evaluate the edge classification ***

yaml_file=cm_m_gnn_eval.yaml
sample='Cosmic w/ Michel' #inferred on
name='signal'
filepath=${base_path}/results/edge_level/${sample_identifier}/
filename_template='cm_m_gnn_'${name}'_'
trained_on='Cosmic w/ Michel + Michel only mixed'
#TODO add code to change the titles of the plots in yaml file.

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@devices.*@devices: ${devices}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file}
sed -i -e "s@sample.*@sample: ${sample}@g" ${yaml_file}
sed -i -e "s@name:.*@name: ${name}@g" ${yaml_file}
sed -i -e "s@filepath.*@filepath: ${filepath}@g" ${yaml_file}
sed -i -e "s@filename_template.*@filename_template: ${filename_template}@g" ${yaml_file}
sed -i -e "s@trained_on.*@trained_on: ${trained_on}@g" ${yaml_file}
echo $'\nRunning evaluation for the edge classification\n'
#acorn eval ${yaml_file} -c ${check_point}

# ============== STAGE: Track Building ==============
# *** infer the track building using a model specified by the checkpoint ***
echo $'\nTrack Building\n'
yaml_file=track_building_infer.yaml
score_cuts="50 55 60 65 70 75 80 85 90 925 950 975 980 985 990 995 999"

input_dir=${stage_dir}
echo Data Split: ${data_split}   
echo Yaml File for the Track Building Stage: ${yaml_file}
echo Input Directory for the Track Building Stage: ${input_dir}
sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file}

for score_cut in ${score_cuts}
do
    stage_dir=${input_dir}/connected_components/${score_cut}/
    if [ "${score_cut}" -le 100 ]; then
        score_cut=$(printf "%.3f" $(bc -l <<<"${score_cut}/100"))
        echo "Score cut: ${score_cut}"
    else
        score_cut=$(printf "%.3f" $(bc -l <<<"${score_cut}/1000"))
        echo "Score cut: ${score_cut}"
    fi

    echo Stage Directory for the Track Building Stage: ${stage_dir}

    sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
    sed -i -e "s@score_cut.*@score_cut: ${score_cut}@g" ${yaml_file}
    acorn infer ${yaml_file} -c ${check_point}
done
cd ..