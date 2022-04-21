############# dominoes ##################

# controller="tdw_physics/target_controllers/dominoes_var.py"
# ARGS_PATH=$HOME"/Downloads/physics-benchmarking-neurips2021/stimuli/generation/configs"
#python $controller @$ARGS_PATH/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /Users/vedanglad/Downloads/example_video --num 200 --height 128 --width 128
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam --num 200 --height 128 --width 128

#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir /mnt/fs4/hsiaoyut/tdw_fewshot/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam --num 200 --height 128 --width 128


# try to add a robot
# controller="tdw_physics/target_controllers/dominoes_var_continue.py"
# ARGS_PATH=$HOME"/Documents/2021/physics-benchmarking-neurips2021-htung/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam_conti --num 200 --height 128 --width 128

############# sliding ##################

controller="tdw_physics/target_controllers/rolling_sliding_var_double.py"
ARGS_PATH=$HOME"/Downloads/physics-benchmarking-neurips2021/stimuli/generation/configs"
#python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam/commandline_args.txt --dir dump/dominoes/pilot_dominoes_2distinct_1middle_tdwroom_fixedcam --num 200 --height 128 --width 128

python3 $controller @$ARGS_PATH/pilot_it2_rollingSliding_simple_ramp_box_2distinct/commandline_args_double.txt --dir /Users/vedanglad/Downloads/example_video/ramp --num 200 --height 128 --width 128

#Workin
# python $controller @$ARGS_PATH/pilot_it2_rollingSliding_simple_ledge_box/commandline_args.txt --dir /Users/vedanglad/Downloads/example_video --num 200 --height 128 --width 128


# controller="tdw_physics/target_controllers/dominoes.py"
# ARGS_PATH=$HOME"/Downloads/physics-benchmarking-neurips2021/stimuli/generation/configs"
# python $controller @$ARGS_PATH/dominoes/pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom/commandline_args.txt --dir /Users/vedanglad/Downloads/example_video --num 200 --height 128 --width 128

