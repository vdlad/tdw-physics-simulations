#!/bin/bash

controller="./tdw_physics/target_controllers/$1.py"
stim_dirs=`ls $2 | grep pilot`
out_dir=$3
gpu=$4

if [[ "$#" = "5" ]]; then
    num="--num $5";
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
    cmd="python $controller @$config --dir $out_dir/$dir --gpu $gpu $num";
    echo $cmd;
    $cmd;
    echo "completeted regeneration of $num $dir stims";
done
