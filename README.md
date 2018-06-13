*work in progress*

# Info

This is an implementation of **image segmentation method** presented in
whitepaper **Active contours driven by local pre-fitting energy for fast image 
segmentation** (Keyan Ding, Linfang Xiao, Guirong Weng,
[researchgate link](https://www.researchgate.net/publication/323237409_Active_contours_driven_by_local_pre-fitting_energy_for_fast_image_segmentation)).


Dependencies:
- OpenCV 3.4

<p align="center"> 
<img src="https://github.com/mmajcher/image-segmentation-method/blob/master/demo.gif">
</p>



# Usage

### Compilation

`make all`

### Available options & params

`./main.out -h`

### Example

`./main.out --initial-contour=initial_contour_ellipse --save-last-image=final_image.jpg --save-final-contours=final_contours.txt --save-all-images-to-dir=some_dir`

### Intial contour file format

See examples: `initial_contour_rectangle` etc.
