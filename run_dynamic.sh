# Go through all folders in a particular folder

set -e

dataset_path=/media/scratch/TUM

for dataset in $(ls $dataset_path)
    do
    #for no_tracking in 3
    for no_tracking in 3 6 9 12 15 
    #for eps in 10 20 30 40 50 60 70
        do
            # If it is a directory then consider
            if [ -d $dataset_path/$dataset ]
                then
                    # Check if the directory name contains walking or sitting or person
                    # If it does then run the script
                    if [[ $dataset == *"sitting"* ]]
                    #if [[ $dataset == *"walking"* ]] # || [[ $dataset == *"sitting"* ]] || [[ $dataset == *"person"* ]]
                        then
                            echo "Running $dataset"
                            python3 dynamic_to_static.py --seq_dir $dataset_path/$dataset/rgb --eps_value 10 --save-images --save-video --template_value ${no_tracking}
                    fi
            fi
        done    
    done
