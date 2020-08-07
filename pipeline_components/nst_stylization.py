import os
import subprocess


def stylization(frames_path, model_name, img_width, stylization_batch_size):
    stylized_frames_dump_dir = os.path.join(frames_path, os.path.pardir, os.path.pardir, model_name.split('.')[0], 'stylized')
    os.makedirs(stylized_frames_dump_dir, exist_ok=True)

    if len(os.listdir(stylized_frames_dump_dir)) == 0:
        print('*' * 20, 'Frame stylization stage started', '*' * 20)
        stylization_script_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'pytorch-nst-feedforward', 'stylization_script.py')
        try:
            # todo: catch CUDA exception
            subprocess.call(['python', stylization_script_path, '--content_input', frames_path, '--batch_size', str(stylization_batch_size), '--img_width', str(img_width), '--model_name', model_name, '--redirected_output', stylized_frames_dump_dir, '--verbose'])
        except Exception as e:
            print(e)
            print(f'Try using smaller stylization batch size, currently = {stylization_batch_size} images in batch.')
            exit(-1)
    else:
        print('Skipping frame stylization, already done.')

    return {"stylized_frames_path": stylized_frames_dump_dir}