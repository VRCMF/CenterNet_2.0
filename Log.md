# CenterNet_2.0
--------------------------------

Time: 24/07/2019 11:11

Note: Build the repository and upload the file to the repository

Download links: https://drive.google.com/file/d/1SHR4CyBG5af9mmobqxLLNJ0Q1biQminB/view?usp=sharing  (ctdet_2x.pth)
                https://drive.google.com/file/d/1y9N4UqBhFEsW1mW_p2eJhaaVgrLJLwYl/view?usp=sharing  (model_last.pth)
                https://drive.google.com/file/d/1HxptmnTnbftZXCUFgf2J6u2lm1bIj4Nc/view?usp=sharing  (widerface_dataset)

--------------------------------

Time: 24/07/2019 11:28

Note: upload the widerface2coco(train/val/test).py files to the annotations folder.

--------------------------------

Time: 24/07/2019 17:59

Note: The "class_num" is wrong(datasets/dataset/widerface.py), it should be 1 not 2(because in the below, the code has
already add the "background" class).
          The coco_test is totally useless, we would use the evaluation method in SRN and DSFD to deal with the evaluation.

