#! /bin/env bash
# to run: ./split_csv.sh <file_to_split> <output_dir>
# TODO: add header to all the splits
PWD=$(pwd)
echo "Current working directory: $PWD"
lines_per_csv=100000
file_to_split=$1
out_file_prefix=$(basename $file_to_split)
out_file_prefix="${out_file_prefix%.csv.csv}"
output_dir=$2
echo $file_to_split
echo $out_file_prefix
echo $output_dir
mkdir -p $output_dir
split -l $lines_per_csv $file_to_split $out_file_prefix
mv ${out_file_prefix}* ${output_dir}
cd $output_dir
for f in *; do mv "$f" "$f.csv"; done
cd $PWD
