#!/bin/bash

controller="./tdw_physics/target_controllers/$1.py" # first arg is controller name without .py
stim_dirs=`ls $2 | grep pilot` # second arg is where your parent dir of configs is; somewhere in human-physics-benchmarking
# stim_dirs=`ls $2 | grep test`
out_dir=$3 # third arg is where to store the new data
gpu=$4 # fourth arg is which gpu to use

if [[ "$#" = "5" ]]; then
    num="--num $5"; # fifth arg is how many trials to make per config
else
    num=""
fi

echo $stim_dirs
echo $controller
echo $num

for dir in $stim_dirs
do
    echo $dir;
    config="$2/$dir/commandline_args.txt";
    newdir="$dir-redyellow" # by default this will append -redyellow to the new dirnames.
    cmd="python $controller @$config --testing_data_mode --dir $out_dir/$newdir --gpu $gpu $num";
    echo $cmd;
    $cmd;
    echo "completeted regeneration of $num $dir stims";
done
