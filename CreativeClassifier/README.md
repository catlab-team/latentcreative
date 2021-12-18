### Go to root folder
```
cd creativegan/CreativeClassifier
```
### Necessary Data
1. 218k Artbreeder images should be under "data/creative_images_218k" (jpeg)
2. BIGGAN generated fake images should be under "data/biggan_generated_train" and "data/biggan_generated_val" (jpg)

### Data Preperation
1. For Artbreeder data preperation, run the commands below. After it finishes, you should see following files. 
    * "train/anc_creative_frames.txt"
    * "val/anc_creative_frames.txt"
    * "train/anc_zero_frames.txt"
    * "val/anc_zero_frames.txt"
```
cd data
python prepare_data.py
```

2. For the preperation of the BIGGAN generated images, no need to run a command. You just need to place images under "data/biggan_generated_train" and "data/biggan_generated_val" in jpg format.


### Train
``` 
python train.py --config config.yaml 
```

### Inference
``` 
python inference.py --input-folder sample_images --checkpoint best.pth 
```
