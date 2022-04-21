import os, sys, copy, glob
from subprocess import PIPE, STDOUT, DEVNULL
import subprocess
from typing import List, Dict, Tuple
from pathlib import Path
import argparse

default_ffmpeg_args = [
    '-vcodec', 'libx264',
    '-crf', '25',
    '-pix_fmt', 'yuv420p'
]

def pngs_to_mp4(
        filename: str,
        image_stem: str,
        png_dir: Path,
        executable: str = 'ffmpeg',
        framerate: int = 30,
        size: List[int] = [256,256],
        start_frame: int = None,
        end_frame: int = None,
        ffmpeg_args: List[str] = default_ffmpeg_args,
        overwrite: bool = False,
        use_parent_dir: bool = False,
        remove_pngs: bool = False,
        rename_movies: bool = True) -> None:
    """
    Convert a directory of PNGs to an MP4.
    """
    cmd = [executable]

    # framerate
    cmd += ['-r', str(framerate)]

    # format
    cmd += ['-f', 'image2']

    # size
    cmd += ['-s', str(size[0]) + 'x' + str(size[1])]

    # filenames
    cmd += ['-i', '"' + str(Path(png_dir).joinpath(image_stem + '%04d.png')) + '"']

    # all other args
    cmd += ffmpeg_args

    # outfile
    if filename[-4:] != '.mp4':
        filename += '.mp4'
    if use_parent_dir:
        filename = filename.split('/')
        filename = '/'.join(filename[:-1]) + '/' + png_dir.parent.name + '_' + filename[-1]
        print("writing %s" % filename)
    cmd += ['"' + str(filename) + '"']

    make_video = subprocess.Popen(' '.join(cmd), shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = make_video.communicate(input=(b'y' if overwrite else b'N'))

    if remove_pngs:
        rm = subprocess.run('rm '  +str(png_dir).replace(' ','\ ') + '/' + image_stem + '*.png', shell=True)

    return cmd, stdout, stderr

def rename_mp4s(
        stimulus_dir: str,
        remove_dir_prefix: bool = False,
        file_pattern: str = "*_img.mp4"):
    """
    Give each mp4 the name '[PARENT_DIR]_[ORIGINAL_MP4]"
    """
    stimulus_dir = Path(stimulus_dir)
    mp4s = sorted(glob.glob(str(stimulus_dir.joinpath(file_pattern))))

    for path in mp4s:
        if remove_dir_prefix:
            newname = path.split(stimulus_dir.name + '_')[-1]
        else:
            newname = stimulus_dir.name + '_' + path.split('/')[-1]
        newpath = stimulus_dir.joinpath(newname)
        mv = ["mv", path, str(newpath)]
        subprocess.run(' '.join(mv), shell=True) 

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="The directory of mp4s to rename")
    parser.add_argument("--files", type=str, default="*_img.mp4", help="The pattern of files to rename")
    parser.add_argument("--remove_prefix", action="store_true", help="remove the name of the dir as prefix")
    args = parser.parse_args()
    
    rename_mp4s(stimulus_dir=args.dir, remove_dir_prefix=args.remove_prefix, file_pattern=args.files)

