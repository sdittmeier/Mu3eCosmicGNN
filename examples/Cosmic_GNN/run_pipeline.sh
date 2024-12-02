#! /bin/env bash
# ============ STAGE: Data Reading ============
# ***produce csv files from the raw data***
base_path=/mnt/data1/tkar/cosmic_new
cd yaml_files

yaml_file=cosmic_reader_v2.yaml
input_dir=${base_path}/raw_data/split_cm/
input_file=${input_dir}/split_cmaa.csv
stage_dir=${base_path}/feature_store/cosmic_michel_f3/test/
data_split_file=dataset_size_aa.csv

echo 'Yaml File for the Data Reading Stage: ${yaml_file}'
echo 'Input Directory for the Data Reading Stage: ${input_dir}'
echo 'Stage Directory for the Data Reading Stage: ${stage_dir}'

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@input_file.*@input_file: ${input_file}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split_file.*@data_split_file: ${data_split_file}@g" ${yaml_file}


acorn infer ${yaml_file}

# ***procude the feature store with truth graphs***
yaml_file=graph_constr.yaml
data_split=\[$(paste -sd ',' "${input_dir}/${data_split_file}")\]
input_dir=${stage_dir}
stage_dir=${base_path}/fully_connected_graphs/cosmic_michel_f3/test/
sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file}

echo 'Yaml File for the Graph Construction Stage: ${yaml_file}'
echo 'Input Directory for the Data Reading Stage: ${input_dir}'
echo 'Stage Directory for the Data Reading Stage: ${stage_dir}'

acorn infer ${yaml_file}
#================================================================================================

# *** remove weird events ***
cd ..
echo $'\nRemoving weird events\n'
input_dir=${stage_dir}
echo $'Input Directory for the weird events remover in the python script: ${input_dir}\n'
sed -i -e "s@input_dir =.*@input_dir = '${input_dir}'@g" remove_weird_events.py
python remove_weird_events.py

cd yaml_files

# ============== STAGE: Edge Classification ==============
# *** infer the edge classification using a model specified by the checkpoint ***
echo $'\nEdge Classification\n'
check_point='/mnt/data1/karres/cosmics_test/cm_m_mixed_gnn/artifacts/best-b575lgmu-val_loss=0.000902-epoch=90.ckpt'
stage_dir=${base_path}/gnn/cosmic_michel_f3/test/cm_m_mixed_gnn
devices=\[7\]
yaml_file=gnn_infer.yaml

echo 'Yaml File for the Edge Classification Stage: ${yaml_file}'
echo 'Input Directory for the Edge Classification Stage: ${input_dir}'
echo 'Stage Directory for the Edge Classification Stage: ${stage_dir}'

sed -i -e "s@input_dir.*@input_dir: ${input_dir}@g" ${yaml_file}
sed -i -e "s@stage_dir.*@stage_dir: ${stage_dir}@g" ${yaml_file}
sed -i -e "s@data_split.*@data_split: ${data_split}@g" ${yaml_file}
sed -i -e "s@devices.*@devices: ${devices}@g" ${yaml_file}

echo $'Running inference for the edge classification\n'
acorn infer ${yaml_file} -c ${check_point}

# *** evaluate the edge classification ***

yaml_file=cm_m_gnn_eval.yaml
sample='Cosmic w/ Michel' #inferred on
name='signal'
filepath=${base_path}/results/edge_level/
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
acorn eval ${yaml_file} -c ${check_point}

cd ..