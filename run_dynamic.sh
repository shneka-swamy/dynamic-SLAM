# Go through all folders in a particular folder

dataset_path=../Datasets

for dataset in $(ls $dataset_path)
    do
    for eps in 10 20 30 40 50 60 70
        do
            # If it is a directory then consider
            if [ -d $dataset_path/$dataset ]
                then
                    echo "Running $dataset"
                    python3 dynamic_to_static.py --seq_dir Datasets/$dataset/rgb --run_yolact --run_homography --eps_value $eps 
            fi
        done    
    done
