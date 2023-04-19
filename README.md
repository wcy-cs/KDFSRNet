# KDFSRNet
Code for paper "Propagating Facial Prior Knowledge for Multi-Task Learning in Face Super-Resolution"



![image](https://user-images.githubusercontent.com/39185517/172369908-4b9698b9-bd92-4158-8a4b-ec05100c13f8.png)

## Requirements
Pytorch 1.8.0, Cuda 10

## Citation 
```
@article{kdfsrnet,
  title={Propagating Facial Prior Knowledge for Multi-Task Learning in Face Super-Resolution},  
  author={Chenyang Wang, Junjun Jiang, Senior Member, IEEE, Zhiwei Zhong and Xianming Liu},
  journal={IEEE Trans. Circuits and Systems for Video Technology},
  year={2022},
  volume={32},
  number={11},
  pages={7317-7331},
  doi={10.1109/TCSVT.2022.3181828}}
}
```

## Results
 [BaiDu](https://pan.baidu.com/s/1bvyPiAnGu_dFI-HfEjn0sw) passward: mji2
 
## Train 
The training phase of our model contains two steps: 1) train the Teacher network with the ground truth; 2) train the Student network with prior knowledge distilated from the Teacher.
1) Train the Teacher Network.
```Python
python train_teacher.py --dir_data data_path  --writer_name Teacher
```
2) Train the Student Network.
```Python
python train_student.py --dir_data data_path  --writer_name Student --teacher_load pretrained_teacher_path
```
## Test
```Python
python test.py --dir_data data_path --load pretrained_model_path 
```
