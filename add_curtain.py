#add curtain
import skvideo.io
import os
import numpy as np
import imageio
from PIL import Image

def load_video(filename):
    #vid = imageio.get_reader(filename,  'ffmpeg')
    return skvideo.io.vread(filename)


#trial_name = "dump/dominoes/pilot_dominoes_2distinct_0middle_tdwroom_fixedcam"
trial_name = "dump/roll/pilot_it2_rollingSliding_simple_ramp_box_2distinct"
trial_id = 0

seq_ids = [0, 1]

data = []
for seq_id in seq_ids:
    filename = os.path.join(trial_name, f"{trial_id:04}_{seq_id:03}_img.mp4")

    print(filename)
    data.append(load_video(filename))


data1 = data[0]
data2 = data[1]

_, H, W, _ = data1.shape
curtain_img = Image.open("curtain.png")
curtain_img = curtain_img.resize((H, W), Image.ANTIALIAS)
curtain_img = np.array(curtain_img.convert("RGB"))


#adding curtain frame
duration = 10
data1_tiled = np.tile(data1[-1:], [duration, 1, 1, 1])
N, H, W, _ = data1_tiled.shape

for t in range(duration):
    h_range = int((t + 1) * H * (1/duration))
    data1_tiled[t, :h_range, :, :] = curtain_img[-h_range,:, :]

curtain_remain = np.tile(np.expand_dims(curtain_img, 0), [3, 1, 1, 1])

data2_tiled = np.tile(data2[:1], [duration, 1, 1, 1])
for t in range(duration):
    h_range = int((duration - t) * H * (1/duration))
    data2_tiled[t, :h_range, :, :] = curtain_img[-h_range:,:, :]



data = np.concatenate([data1, data1_tiled, curtain_remain, data2_tiled, data2], 0)

skvideo.io.vwrite("outputvideo.mp4", data)

#writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
#for i in xrange(5):
#        writer.writeFrame(outputdata[i, :, :, :])
#writer.close()


