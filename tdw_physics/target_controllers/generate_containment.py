import os
import numpy as np
import pandas as pd

'''
Script to generate containment controller under variety fo conditions
'''

#To generate catch stims, use 1 block and 0999 as the seed
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
	#parser.add_argument('--nIter', type=int, help='how many bootstrap iterations?', default=1000)
    args = parser.parse_args()

    target_scales = [0.1,0.5]#np.arange(0.1,1,0.4)
    container_scales = [0.8] #np.arange(0.1,1,0.4)
    probe_mass = [1,3]
    probe_scales = [0.1]
    force_scales = [4]
    print('Now running ...')
    for tscale in target_scales:
        for mscale in container_scales:
            for pmass in probe_mass:
                for pscale in probe_scales:
                    for fscale in force_scales:
                        cmd_string = 'python containment.py --save_passes "_img" --save_movies --dir /Users/choldawa/Documents/Projects/tdw_physics/tdw_physics/target_controllers/stimuli \
                                                                            --num={} \
                                                                            --random={}\
                                                                            --tscale={} \
                                                                            --mscale={} \
                                                                            --pmass={}\
                                                                            --pscale={}\
                                                                            --fscale={}\
                                                                            '.format(1,
                                                                            1,
                                                                            tscale,
                                                                            mscale,
                                                                            pmass,
                                                                            pscale,
                                                                            fscale)
                        print(cmd_string)
                        os.system(cmd_string)
					#thread.start_new_thread(os.system,(cmd_string,))
