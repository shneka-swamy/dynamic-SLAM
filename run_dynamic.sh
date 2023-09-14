# Go through all folders in a particular folder

set -e

dataset_path=/media/scratch/TUM

for dataset in $(ls $dataset_path)
    do
    for eps in 10 20 30 40 50 60 70
        do
            # If it is a directory then consider
            if [ -d $dataset_path/$dataset ]
                then
                    # Check if the directory name contains walking or sitting or person
                    # If it does then run the script

                    if [[ $dataset == *"walking"* ]] || [[ $dataset == *"sitting"* ]] || [[ $dataset == *"person"* ]]
                        then
                            echo "Running $dataset"
                            python3 dynamic_to_static.py --seq_dir $dataset_path/$dataset/rgb --run_yolact --run_homography --eps_value $eps 
                    fi
            fi
        done    
    done
