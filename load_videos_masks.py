import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from util import save
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize

warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')

# def trasncode_for_ui(input_path, output_path):
#     fill_command = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "warning", "-an",
#                     "-i", input_path,
#                     "-vcodec", "libx264", "-pix_fmt", "yuv420p",
#                     "-profile:v", "baseline", "-level", "3", "-b:v", "2M",
#                     output_path]
#     proc = subprocess.run(" ".join(fill_command), shell=True)
    

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path


def process_video(input_video, output_video, start, length, width, height, x, y):
    filter_command = f'"[0]pad=w=512+iw:h=512+ih:x=256:y=256:color=gray,crop={width}:{height}:{x}:{y},fps=fps=31.25"'
    
    command = ["ffmpeg -hide_banner -y -loglevel warning",
               "-i", input_video,
               "-ss", start,
               "-t", length,
               "-filter_complex", filter_command,
               "-vcodec", "libx264", "-pix_fmt", "yuv420p",
               "-profile:v", "baseline", "-level", "4",
               output_video]
    subprocess.run(" ".join(command), shell=True)


def second_to_ffmpeg_time(val):
    hours = val // 3600
    seconds = val % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    
    hours = str(int(hours)).zfill(2)
    minutes = str(int(minutes)).zfill(2)
    
    milliseconds = str(round(seconds % 1, 2))[2:]
    seconds = str(int(seconds)).zfill(2)
    
    return f"{hours}:{minutes}:{seconds}.{milliseconds}"


def run(data):
    video_id, args = data
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       return
       download(video_id.split('#')[0], args)

    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       print ('Can not load video %s, broken link' % video_id.split('#')[0])
       return 
    input_video_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    reader = imageio.get_reader(input_video_path)
    fps = reader.get_meta_data()['fps']
    
    for frame in reader:
        break
    reader.close()

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]
    
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[]} for j in range(df.shape[0])]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]
    
    try:
        for entry in all_chunks_dict:
            start_time = entry['start'] / ref_fps
            end_time = entry['end'] / ref_fps

            start_ffmpeg = second_to_ffmpeg_time(start_time)
            length_ffmpeg = second_to_ffmpeg_time(end_time - start_time)

            input_video_name = str(input_video_path)

            if 'person_id' in df:
                first_part = df['person_id'].iloc[0] + "#"
            else:
                first_part = ""
            first_part = first_part + '#'.join(video_id.split('#')[::-1])
            path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
            output_path = os.path.join(args.out_folder, partition, path)
            output_video_name = str(output_path)

            left, top, right, bot = entry['bbox']
            left = int(left / (ref_width / frame.shape[1]))
            top = int(top / (ref_height / frame.shape[0]))
            right = int(right / (ref_width / frame.shape[1]))
            bot = int(bot / (ref_height / frame.shape[0]))

            height = bot - top
            width = right - left
            max_size = max(height, width)
            min_size = min(height, width)

            x = 256 + left - (max_size - width)  // 2
            y = 256 + top  - (max_size - height) // 2
            
#            print(x, y, max_size, width, height)

            x = int(x - max_size * 0.1)
            y = int(y - max_size * 0.1)
            max_size = int(1.2 * max_size)
            
            new_x = x - max_size // 4
            new_max_size = int(max_size * 1.5)
#            print(new_x, new_x + new_max_size, y, y + new_max_size, frame.shape[0] + 256, frame.shape[1] + 256)
            
#             if x + new_max_size < frame.shape[1] and y + new_max_size < frame.shape[0]:
#                 print()
#                 print('OK')

#             print()
#             print('X:', x, 'Y:', y, 'SIZE:', max_size)
#             print('width:', frame.shape[1], 'height:', frame.shape[0])
#             print('new X:', new_x, 'new_max_size:', new_max_size)

            result_mask = np.zeros((frame.shape[0] + 512, frame.shape[1] + 512))
            result_mask[256:-256, 256:-256] = 1
        
            crop_mask = np.zeros((frame.shape[0] + 512, frame.shape[1] + 512))
            crop_mask[y:y+new_max_size, new_x:new_x+new_max_size] = 1
            
            intersection_mask = result_mask * crop_mask
            intersection_size = intersection_mask.sum()
            
            print(output_video_name, intersection_size / crop_mask.sum())
        
        

#             wholesize = new_max_size * new_max_size
#             cropsize_w = new_max_size
#             if new_x < 256:
#                 cropsize_w = cropsize_w - (256 - new_x)
#             if new_x + new_max_size > (frame.shape[1] - 256):
#                 cropsize_w = cropsize_w - ((new_x + new_max_size) - (frame.shape[1] - 256))
#             cropsize_h = new_max_size
#             if y < 256:
#                 cropsize_h = cropsize_h - (256 - y)
#             if y + new_max_size > (frame.shape[0] - 256):
#                 cropsize_h = cropsize_h - ((y + new_max_size) - (frame.shape[0] - 256))
                
#             cropsize = cropsize_w * cropsize_h
#             print(new_x, y, cropsize_w, cropsize_h, new_max_size)
#             gray_part_size = wholesize - cropsize
            
#             print(gray_part_size / wholesize)
            
            
#            process_video(input_video_name, output_video_name, start_ffmpeg, length_ffmpeg, new_max_size, new_max_size, new_x, y)
    except Exception as e:
        print(e)
        print('FAILED', input_video_name)
        
#    print('FINISHED', input_video_name)
#    os.remove(input_video_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='taichi-metadata-new.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl', help='Path to youtube-dl')
 
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in pool.imap_unordered(run, zip(video_ids, args_list)):
        None  
