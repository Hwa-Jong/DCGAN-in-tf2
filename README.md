# DCGAN-in-tf2
###### DCGAN in tensorflow1 : [link](https://github.com/Hwa-Jong/DCGAN-in-tf1)
###### Tensorflow2 implementation of DCGAN.
###### [paper](https://arxiv.org/pdf/1511.06434.pdf)
----------------
## Prerequisites
- Python 3.8
- Tensorflow 2.4.1
- Opencv-contrib-python 4.5.1.48

----------------
## Generator architecture of DCGAN

![]()

----------------
## Usage

1. Preparing your dataset. (In my case, I prepared images in the "dataset" directory.)

2. Train the model.
> ```
> python3 main.py --dataset_dir=<your dataset path>
> ```
> ex)
> ```
> python3 main.py --dataset_dir=dataset --epochs=100(default) --batch_size=128(default)
> ```
3. Using pre-trained model
> ```
> python3 main.py --dataset_dir=<your dataset path> --load_path=<model path>
> ```
> ex)
> ```
> python3 main.py --dataset_dir=dataset --epochs=100(default) --batch_size=128(default) --load_path=results/0001_DCGAN_batch-128_epoch-100/ckpt/model.ckpt-50
> ```
4. Generate images
> ```
> python3 generate.py --load_path=<model path>
> ```
> ex)
> ```
> python3 generate.py --load_path=results/0001_DCGAN_batch-128_epoch-100/ckpt/model.ckpt-50 --generate_num=16(default) --seed=22222(default)
> ```

----------------
## Result 
> ### 1 epoch
> ![]()

> ### 10 epochs
> ![]()

> ### 50 epochs
> ![]()

> ### 80 epochs
> ![]()

> ### 100 epochs
> ![]()


----------------
## Note..
###### I found a strange phenomenon while using tf2.
###### I found a significant increase in error at about 70 ~ 85 epochs(in this case, 86 epochs). Also, the result is not good. But after a few epochs, the results got better again.
###### It was not found in tf1. I'm going to check what's wrong.

> ### 85 epoch
> ![]()
> ### 86 epoch
> ![]()
> ### 87 epoch
> ![]()
> ### 92 epoch
> ![]()
> ### 93 epoch
> ![]()

