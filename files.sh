# get profile files from the grid archive
rm -rf profiles_data
mkdir profiles_data

for file in grid_archive_run/models/*; do
    echo $file
    NUMBER=$(echo $file | tr -dc '0-9') ; echo $NUMBER
    work_dir="grid_archive_run/models/work_$NUMBER"
    tar -zxf $file -C grid_archive_run/models/
    mkdir profiles_data/model_$NUMBER
    cp $work_dir/LOGS/profile*.data profiles_data/model_$NUMBER
    tar -czvf profiles_data/model_$NUMBER.tar.gz profiles_data/model_$NUMBER
    rm -rf profiles_data/model_$NUMBER
    rm -rf $work_dir
done

## change names of the files
for file in archive/profiles_data/*; do
    echo $file
    NUMBER=$(echo $file | tr -dc '0-9') ; echo $NUMBER
    mv $file archive/profiles_data/profiles_track_$NUMBER.tar.gz
done