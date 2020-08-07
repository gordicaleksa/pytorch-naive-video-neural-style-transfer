## Naive Video Fast NST :movie_camera: + :zap::computer: + :art: = :heart:
This repo is a wrapper around [my implementation of fast NST](https://github.com/gordicaleksa/pytorch-nst-feedforward) (for static images) and additionally provides:
1. Support for creating (naive - no temporal loss included) videos
2. Support for creating segmentation masks for the person talking

It's an accompanying repo for [this video series on YouTube](https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608).

<p align="left">
<a href="https://www.youtube.com/watch?v=S78LQebx6jo" target="_blank"><img src="https://img.youtube.com/vi/S78LQebx6jo/0.jpg" 
alt="NST Intro" width="480" height="360" border="10" /></a>
</p>

The first video of the series was created exactly using this method (I also used ReCoNet for 1 part).

## Combining stylized frames with original frames (via seg masks)

<p align="center">
<img src="data/examples/stylized.jpg" width="270px">
<img src="data/examples/person_masked.jpg" width="270px">
<img src="data/examples/bkg_masked.jpg" width="270px">
</p>

And these were produced using this segmentation mask:

<p align="center">
<img src="data/examples/mask.png" width="400px">
</p>

It's not perfect but this one was created in a fully automatic fashion.

## Combining 2 types of stylized frames (via seg masks)

Similarly...

## Setup

1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `activate pytorch-nst-fast` (if you want to run scripts from your console otherwise set the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

-----

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system. 

Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md) and use the most up-to-date versions of Miniconda and CUDA/cuDNN (I recommend CUDA 10.1 or 10.2 as those are compatible with PyTorch 1.5, which is used in this repo, and newest compatible cuDNN).

## Usage

# Debugging
Q: My style/content loss curves just spiked in the middle of training?<br/>
A: 2 options: a) rerun the training (optimizer got into a bad state) b) if that doesn't work lower your style weight

<p align="left">
<img src="data/examples/readme_pics/spike.png" width="683"/>
</p>

Q: How can I see the exact parameters that you used to train your models?<br/>
A: Just run the model in the `stylization_script.py`, training metadata will be printed out to the output console.

# todo: add GIFs to readme

## Citation

If you find this code useful for your research, please cite the following:

```
@misc{Gordić2020-naive-video-nst,
  author = {Gordić, Aleksa},
  title = {pytorch-naive-video-nst},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-naive-video-nst}},
}
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-naive-video-nst/blob/master/LICENCE)