import cv2
import os
import numpy as np
import shutil
from os.path import isfile, join
from tools.infer.predict_system import main
import argparse
TEMP_FOLDER = 'temp'
INFERENCE_FOLDER = 'inference_results'

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def video_to_image(video_file, output_dir):
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)

    while success:
        output_images = os.path.join(output_dir, "frame%d.png")
        cv2.imwrite(output_images % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return fps


def image_to_video(input_photos, output_video, fps):
    pathIn = input_photos
    pathOut = output_video
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        print('filename', filename)

        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(pathOut, fourcc, float(fps), size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--video_file", type=str)
    parser.add_argument("--output_video", type=str, default='output.mp4')

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str, default=TEMP_FOLDER)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", default="./inference/ch_ppocr_server_v2.0_det_infer/")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    args = parser.parse_args()
    # output
    output_dir = TEMP_FOLDER
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_photos = INFERENCE_FOLDER
    if not os.path.exists(input_photos):
        os.makedirs(input_photos)
    fps = video_to_image(args.video_file, output_dir)
    main(parser.parse_args())
    image_to_video(input_photos, args.output_video, fps)
    shutil.rmtree(TEMP_FOLDER)
    shutil.rmtree(INFERENCE_FOLDER)