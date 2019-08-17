# congeneric_renderer

[Usage]

        1. Start with the .ini file in the /outcome folder.
        2. In the 'train' mode, you can train the model with the self-play manner to train the renderer, segmentor and all discriminators.
        3. Training images should have size 320x320.
        4. In the 'test' mode, you can then test an image by 'training' all the models in an on-line and weakly supervised way. 
        5. The 'test' mode will firstly render the whole training dataset for a more uniform intensity distribution, as we describe in the MICCAI paper.

10 test prediction results along the 25 iterations are attached in the results folder for your reference. Segmentation refinement can be observed as the iteration increases.
Currently, this method only works on two fetal head datasets, and doesn't present advantages on other more complex tasks.

Improvement curve of DICE along 25 iterations.
![image](https://github.com/xy0806/congeneric_renderer/blob/master/test_results/curve/1_dice.png)
![image](https://github.com/xy0806/congeneric_renderer/blob/master/test_results/curve/2_dice.png)
![image](https://github.com/xy0806/congeneric_renderer/blob/master/test_results/curve/3_dice.png)

Improved segmentation results along 25 iterations.
![image](https://github.com/xy0806/congeneric_renderer/blob/master/iter_25.png)


If the code is helpful for your research, please cite our code as:

    @inproceedings{yang2018generalizing,
      title={Generalizing deep models for ultrasound image segmentation},
      author={Yang, Xin and Dou, Haoran and Li, Ran and Wang, Xu and Bian, Cheng and Li, Shengli and Ni, Dong and Heng, Pheng-Ann},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={497--505},
      year={2018},
      organization={Springer}
    }

