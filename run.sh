#!/bin/bash
#SBATCH --account=yuni0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=dotatate_all_patients_group1
#SBATCH --mail-user=jiayx@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --time=32:00:00
#SBATCH --mem=128g

module load python3.9-anaconda/2021.11
export PYTHONPATH=$PWD:$PYTHONPATH

# Define the absolute path to the base directory
BASEDIR="/home/jiayx/ondemand/data/sys/myjobs/projects/default/sperf/dotatate_patients_256_nearest/group1"
echo "BASEDIR set to: $BASEDIR"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/pkgs/arc/python3.9-anaconda/2021.11/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh" ]; then
        . "/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/sw/pkgs/arc/python3.9-anaconda/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate sperf

for dir in */; do
    # Skip if not a directory
    if [[ ! -d "$dir" ]]; then continue; fi
    
    # Define config directory
    config_dir="${BASEDIR}/${dir}/config"
    src_dir="${BASEDIR}/${dir}/src"

    # Check for required files
    required_files=("params.txt" "params_scatter.txt")
    for file in "${required_files[@]}"; do
        if [[ ! -f "${config_dir}/$file" ]]; then
            echo "Required file $file not found in $config_dir"
            exit 1
        fi
    done

    # Edit the files
    for file in "${required_files[@]}"; do
        file_path="${config_dir}/${file}"
        base_name="${dir%/}" # Removes trailing slash

        # Determine new values (patient names, etc.) based on the file name
        case "$file" in
            "params_scatter.txt")
                new_expname="${base_name}_scatter"
                new_datadir="${config_dir}/../patient_data/proj/proj_${base_name}_256_nearest_scatter.jld2"
                new_psfdir="${config_dir}/../patient_data/radial_position/radial_position_proj_${base_name}.txt"
                ;;
            "params.txt")
                new_expname="${base_name}"
                new_datadir="${config_dir}/../patient_data/proj/proj_${base_name}_256_nearest.jld2"
                new_psfdir="${config_dir}/../patient_data/radial_position/radial_position_proj_${base_name}.txt"
                ;;
        esac

        # Edit the file
        sed -i "s/^expname = .*/expname = ${new_expname}/" "$file_path"
        sed -i "s|^datadir = .*|datadir = ${new_datadir}|" "$file_path"
        sed -i "s|^psfdir = .*|psfdir = ${new_psfdir}|" "$file_path"

        # Check if main.py exists in the src directory
        if [[ ! -f "${src_dir}/main.py" ]]; then
            echo "main.py not found in ${src_dir}"
            continue # Skip to next iteration
        fi

        # Execute the Python script with the absolute path to the config file
        (cd "$src_dir" && python main.py --config "$file_path")
        wait
    done
done


conda deactivate
