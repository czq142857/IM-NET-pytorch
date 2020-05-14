# IM-NET-pytorch
PyTorch 1.2 implementation for paper "Learning Implicit Fields for Generative Shape Modeling", [Zhiqin Chen](https://www.sfu.ca/~zhiqinc/),  [Hao (Richard) Zhang](https://www.cs.sfu.ca/~haoz/).

### [Project page](https://www.sfu.ca/~zhiqinc/imgan/Readme.html) |   [Paper](https://arxiv.org/abs/1812.02822)

### [Original implementation](https://github.com/czq142857/implicit-decoder)

### [Improved TensorFlow1 implementation](https://github.com/czq142857/IM-NET)


## Improvements

In short, this repo is an implementation of [IM-NET](https://github.com/czq142857/IM-NET) with the framework provided by [BSP-NET-pytorch](https://github.com/czq142857/BSP-NET-pytorch).

The improvements over the [original implementation](https://github.com/czq142857/implicit-decoder) is the same as [IM-NET (improved TensorFlow1 implementation)](https://github.com/czq142857/IM-NET):

Encoder:

- In IM-AE (autoencoder), changed batch normalization to instance normalization.

Decoder (=generator):

- Changed the first layer from 2048-1024 to 1024-1024-1024.
- Changed latent code size from 128 to 256.
- Removed all skip connections.
- Changed the last activation function from sigmoid to clip ( max(min(h, 1), 0) ).

Training:

- Trained one model on the 13 ShapeNet categories as most Single-View Reconstruction networks do.
- For each category, sort the object names and use the first 80% as training set, the rest as testing set, same as [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet).
- Reduced the number of sampled points by half in the training set. Points were sampled on 256<sup>3</sup> voxels.
- Removed data augmentation (image crops), same as [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).
- Added coarse-to-fine sampling for inference to speed up testing.
- Added post-processing to make the output mesh smoother. To enable, find and uncomment all *"self.optimize_mesh(vertices,model_z)"*.


## Citation
If you find our work useful in your research, please consider citing:

	@article{chen2018implicit_decoder,
	  title={Learning Implicit Fields for Generative Shape Modeling},
	  author={Chen, Zhiqin and Zhang, Hao},
	  journal={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2019}
	}

## Dependencies
Requirements:
- Python 3.5 with numpy, scipy and h5py
- [PyTorch 1.2](https://pytorch.org/get-started/locally/)
- [PyMCubes](https://github.com/pmneila/PyMCubes) (for marching cubes)

Our code has been tested on Ubuntu 16.04 and Windows 10.


## Datasets and pre-trained weights
The original voxel models are from [HSP](https://github.com/chaene/hsp).

The rendered views are from [3D-R2N2](https://github.com/chrischoy/3D-R2N2).

Since our network takes point-value pairs, the voxel models require further sampling.

For data preparation, please see directory [point_sampling](https://github.com/czq142857/IM-NET/tree/master/point_sampling).

We provide the ready-to-use datasets in hdf5 format, together with our pre-trained network weights.

- [IM-NET-pytorch](https://drive.google.com/open?id=1ykE6MB2iW1Dk5t4wRx85MgpggeoyAqu3)

Backup links:

- [IM-NET-pytorch](https://pan.baidu.com/s/10695F20-xTWCrltYGhBPcQ) (pwd: bqex)


## Usage

Please use the provided scripts *train_ae.sh*, *train_svr.sh*, *test_ae.sh*, *test_svr.sh* to train the network on the training set and get output meshes for the testing set.

To train an autoencoder, use the following commands for progressive training. 
```
python main.py --ae --train --epoch 200 --sample_dir samples/all_vox256_img0_16 --sample_vox_size 16
python main.py --ae --train --epoch 200 --sample_dir samples/all_vox256_img0_32 --sample_vox_size 32
python main.py --ae --train --epoch 200 --sample_dir samples/all_vox256_img0_64 --sample_vox_size 64
```
The above commands will train the AE model 200 epochs in 16<sup>3</sup> resolution, then 200 epochs in 32<sup>3</sup> resolution, and finally 200 epochs in 64<sup>3</sup> resolution.
Training on the 13 ShapeNet categories takes about 3 days on one GeForce RTX 2080 Ti GPU.

After training, you may visualize some results from the testing set.
```
python main.py --ae --sample_dir samples/im_ae_out --start 0 --end 16
```
You can specify the start and end indices of the shapes by *--start* and *--end*.


To train the network for single-view reconstruction, after training the autoencoder, use the following command to extract the latent codes:
```
python main.py --ae --getz
```
Then use the following commands to train the SVR model:
```
python main.py --svr --train --epoch 1000 --sample_dir samples/all_vox256_img1
```
After training, you may visualize some results from the testing set.
```
python main.py --svr --sample_dir samples/im_svr_out --start 0 --end 16
```


## License
This project is licensed under the terms of the MIT license (see LICENSE for details).


