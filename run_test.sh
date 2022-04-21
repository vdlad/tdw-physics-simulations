#!/bin/bash


#controller="./tdw_physics/target_controllers/dominoes.py"
controller="./tdw_physics/target_controllers/cloth_sagging.py"
#config="/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-drop/pilot_it2_drop_all_bowls_box/commandline_args.txt"
#config="/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom/commandline_args.txt"
config="/home/htung/Documents/2021/human-physics-benchmarking/stimuli/generation/pilot-clothSagging/test10/commandline_args.txt"

cmd="python $controller @$config --dir data/containment/pilot-drop --save_meshes  --num 10 --height 128 --width 128 --port 1071"
echo $cmd

$cmd;