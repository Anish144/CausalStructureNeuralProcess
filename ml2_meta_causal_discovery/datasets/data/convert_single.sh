# for i in 0,20,0 20,40,1 40,60,2 80,100,3 100,120,4 120,140,5 140,160,6 160,180,7 180,200,8 200,220,9
for i in 20,40,1 40,60,2
do
    IFS=",";
    set -- $i;
    python convert_into_single_files.py --data_start=$1 --data_end=$2 --file_index=$3
done