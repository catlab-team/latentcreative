import numpy as np
from scipy.stats import truncnorm
import PIL.ImageDraw
import PIL.ImageFont

import cv2
import os.path as osp

class VideoWriter(object):
    """
    Video writer which handles video recording overhead
    Usage:
        object creation: provide path to write
        write:
        release:
    """
    def __init__(self, video_file, fps=25, scale=1.0):
        """
        :param video_file: path to write video. Perform nothing in case of None
        :param fps: frame per second
        :param scale: resize scale
        """
        self.video_file = video_file
        self.fps = fps
        self.writer = None
        self.scale = scale

    def write(self, frame):
        """
        :param frame: numpy array, (H, W, 3), BGR, frame to write
        :return:
        """
        h, w = frame.shape[:2]
        h_rsz, w_rsz = int(h * self.scale), int(w * self.scale)
        frame = cv2.resize(frame, (w_rsz, h_rsz))
        if self.writer is None:
            video_dir = osp.dirname(osp.realpath(self.video_file))
            if not osp.exists(video_dir):
                os.makedirs(video_dir)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.video_file, fourcc, self.fps,
                                          tuple(frame.shape[1::-1]))
        self.writer.write(frame)

    def release(self):
        """
        Manually release
        :return:
        """
        if self.writer is None:
            return
        self.writer.release()

    def __del__(self):
        self.release()


def interpolate_frames(frame_list, num_interpolated_bw_frames):
    interpolated_frames = []
    for frame_idx in range(len(frame_list)-1):
        frame_1 = frame_list[frame_idx]
        frame_2 = frame_list[frame_idx+1]
        for i in range(num_interpolated_bw_frames):
            alpha = i/num_interpolated_bw_frames
            interpolated_frame = frame_1*(1-alpha) + frame_2*(alpha)
            interpolated_frame = np.uint8(interpolated_frame)
            interpolated_frames.append(interpolated_frame)          
    interpolated_frames.append(frame_list[-1])
    return interpolated_frames


def truncated_z_sample(batch_size, dim_z, truncation=1):
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z))
    return truncation * values

def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        imarray = np.uint8(imarray)
        # raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows * H, cols * W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def annotate_outscore(array, outscore):
    for i in range(array.shape[0]):
        I = PIL.Image.fromarray(np.uint8(array[i,:,:,:]))
        draw = PIL.ImageDraw.Draw(I)
        font =  PIL.ImageFont.truetype("./utils/arial.ttf", int(array.shape[1]/8.5))
        #print("outscore", outscore)
        #message = str(round(np.squeeze(outscore)[i], 2))
        message = str(round(outscore[i][0], 2))
        
        #message = str(round(float(outscore), 2))
        x, y = (0, 0)
        w, h = font.getsize(message)
        #print(w, h)
        draw.rectangle((x, y, x + w, y + h), fill='white')
        draw.text((x, y), message, fill="black", font=font)
        array[i, :, :, :] = np.array(I)
    return(array)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
