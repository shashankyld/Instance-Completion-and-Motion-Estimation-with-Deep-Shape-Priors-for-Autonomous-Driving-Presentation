
# Mile Stones

![WhatsApp Image 2023-04-16 at 15.09.31.jpg](Mile%20Stones%20d877b8adcbdc41ffa4da8cad2e0e8849/WhatsApp_Image_2023-04-16_at_15.09.31.jpg)

### M1: Basic knowledge learning and paper reading (~4 weeks)

- Pytorch learning: finish [[A1]](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment1.html) [[A2]](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment2.html) [[A3]](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment3.html) Assignments of [eecs498](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/). (Only if you know nothing about Pytorch before)
- Read and understand the following papers:
    - Cluster-VO [[code](https://drive.google.com/file/d/1zQr11Ne_52HTXcIHRH5rsbfuhi4B8HJZ/view)][[paper](https://arxiv.org/abs/2003.12980)][[video](https://www.youtube.com/watch?v=paK-WCQpX-Y)]
    - DeepSDF [[code](https://github.com/facebookresearch/DeepSDF)][[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)]
    - Nerf [[paper](https://www.matthewtancik.com/nerf)]
    - DSP-SLAM [[code](https://github.com/JingwenWang95/DSP-SLAM)][[paper](https://arxiv.org/abs/2108.09481)][[video](https://www.youtube.com/watch?v=of4ANH24LP4)]
    - Weakly Supervised Learning of Rigid 3D Scene Flow [[code](https://github.com/zgojcic/Rigid3DSceneFlow)][[paper](https://arxiv.org/abs/2102.08945)]
- Get familiar with the following datasets we will use:
1. [KITTI](https://www.cvlibs.net/datasets/kitti/)
2. [Nuscenes](https://www.nuscenes.org/)
3. [Waymo](https://waymo.com/open/)

### M2: Code Reading and demo test (~2 weeks)

- Read the code of DSP-SLAM and test it in different datasets, especially in dynamic scenes, and observe failure situations.
- Read the instance association part of Cluster-VO.

### M3: Build the main program framework (~4 weeks)

- Finish the data preprocessing part, including detection network and lidar odometry.
- Finish 2 in Methodology based on DSP-SLAM.
- Finish 3 in Methodology with the simplest method (bounding box overlap).

### M4: Make some contribution (~6 weeks)

- Finish 4 in Methodology by designing a proper loss function.
- Explore more robust methods to achieve instance association.
- Explore the complete differentiable process from instance association to motion estimation.

### M5: Final Result: Evaluation (~4 weeks)

- Use scene flow dataset and metric to evaluate motion estimation.
- Design another metric to evaluate instance association with panoptic segmentation dataset.
