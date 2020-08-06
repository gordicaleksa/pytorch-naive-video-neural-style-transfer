import os
import shutil
import subprocess


def create_videos(video_metadata, relevant_directories, frame_name_format, delete_source_imagery):
    stylized_frames_path = relevant_directories['stylized_frames_path']
    dump_path_bkg_masked = relevant_directories['dump_path_bkg_masked']
    dump_path_person_masked = relevant_directories['dump_path_person_masked']

    combined_img_bkg_pattern = os.path.join(dump_path_bkg_masked, frame_name_format)
    combined_img_person_pattern = os.path.join(dump_path_person_masked, frame_name_format)
    stylized_frame_pattern = os.path.join(stylized_frames_path, frame_name_format)

    dump_path = os.path.join(stylized_frames_path, os.path.pardir)
    combined_img_bkg_video_path = os.path.join(dump_path, f'{os.path.basename(dump_path_bkg_masked)}.mp4')
    combined_img_person_video_path = os.path.join(dump_path, f'{os.path.basename(dump_path_person_masked)}.mp4')
    stylized_frame_video_path = os.path.join(dump_path, 'stylized.mp4')

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        audio_path = relevant_directories['audio_path']

        def build_ffmpeg_call(img_pattern, audio_path, out_video_path):
            input_options = ['-r', str(video_metadata['fps']), '-i', img_pattern, '-i', audio_path]
            encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy']
            pad_options = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']  # libx264 won't work for odd dimensions
            return [ffmpeg] + input_options + encoding_options + pad_options + [out_video_path]

        subprocess.call(build_ffmpeg_call(combined_img_bkg_pattern, audio_path, combined_img_bkg_video_path))
        subprocess.call(build_ffmpeg_call(combined_img_person_pattern, audio_path, combined_img_person_video_path))
        subprocess.call(build_ffmpeg_call(stylized_frame_pattern, audio_path, stylized_frame_video_path))
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')

    if delete_source_imagery:
        shutil.rmtree(dump_path_bkg_masked)
        shutil.rmtree(dump_path_person_masked)
        shutil.rmtree(stylized_frames_path)
        print('Deleting stylized/combined source images done.')