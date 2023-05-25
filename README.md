# Streaming Audio Transformers for Online Audio Tagging

Highlights:

* Transformers capable of being used for online-audio tagging. The model processes at most **2s at a time** and incorporated past predictions. Best used when **deployed on stationary devices** (cameras, speakers, electronic household items). 
* Different from most research, SAT is aimed to **deploy an audio tagger**, not use it as a feature extractor to some other task.
* Performance: 45.1 mAP on the best model with 2s delay, and 43.3 mAP on a ViT-Tiny.
* Memory and computational footprint is manageable. Our SAT-T can be **easily deployed on mobile devices**, with 20 Mb (size) parameters and 9 Mb (float32) RAM. For 1s inference, we only require 4.3 Mb of RAM.
* Partially solves most AT problems that deploy transformers such as: "my transformer's performance is very bad on shorter than 10s sized clips" and "pad the input to 10s and pay the computational overhead price". SAT helps problems like [1](https://github.com/qiuqiangkong/audioset_tagging_cnn/issues/50), [2](https://github.com/fschmid56/EfficientAT/issues/3), [3](https://github.com/YuanGongND/ast/issues/92), [4](https://github.com/YuanGongND/ast/issues/87) and [5](https://github.com/YuanGongND/ast/issues/60).
* SAT can track **long-term events** effectively, whereas standard audio taggers generally have a high score-variance between successive chunks.



|                           | Model  | Streamable? | \#Token | PeakMem | GFlops | mAP  |
|---------------------------|--------|-------|---------|------------|--------|------|
| 2s delay                  | ViT-T  |       | 48      | 7.6 M      | 0.5    | 39.1 |
|                           | ViT-S  |       | 48      | 15 M       | 2.1    | 40.9 |
|                           | ViT-B  |       | 48      | 30 M       | 8.2    | 41.5 |
|                           | AST    |       | 1212    | 2.2 G      | 202    | 39.7 |
|                           | BEATs  |       | 96      | 83 M       | 17.8   | 38.7 |
|                           | HTS-AT |       | 1024    | 171 M      | 42     | 5.2  |
|                           | SAT-T  | Y | 48/48   | 9 M        | 0.5    | 43.3 |
|                           | SAT-S  | Y |         | 18 M       | 2.1    | 43.4 |
|                           | SAT-B  | Y |         | 36 M       | 8.2    | 45.1 |
|---------------------------|--------|-------|---------|------------|--------|------|
| 1 s delay                 | ViT-T  |       | 24      | 3.8 M      | 0.3    | 33.0 |
|                           | ViT-S  |       | 24      | 7.5 M      | 1.1    | 34.9 |
|                           | ViT-B  |       | 24      | 14 M       | 4.1    | 34.2 |
|                           | AST    |       | 1212    | 2.2 G      | 202    | 36.6 |
|                           | BEATs  |       | 48      | 83 M       | 17.8   | 35.2 |
|                           | HTS-AT |       | 128     | 171 M      | 42     | 2.4  |
|                           | SAT-T  | Y | 24/24   | 4.3 M      | 0.3    | 40.1 |
|                           | SAT-S  | Y |         | 9 M        | 1.1    | 40.2 |
|                           | SAT-B  | Y |         | 16 M       | 4.1    | 41.4 |



## Preparation

```bash
git clone https://github.com/RicherMans/SAT
pip3 install -r requirements.txt
```


## Inference


We prepare a simple script to run inference for all proposed models (ViT-T/S/B and SAT-T/S/B).
The checkpoints are hosted on [zenodo](https://zenodo.org/record/7964975).

Running inference for the two water samples seen in the paper:

```bash
python3 inference.py samples/jkLRith2wcc.wav # Default is the SAT-T 2s model
```

Outputs:

```bash
#===== samples/jkLRith2wcc.wav =====
#[0.0s]-[2.0s] Topk-1 Stream                         0.9582
#[0.0s]-[2.0s] Topk-2 Trickle, dribble               0.1661
#[0.0s]-[2.0s] Topk-3 Boat, Water vehicle            0.1254
#
#[2.0s]-[4.0s] Topk-1 Stream                         0.8668
#[2.0s]-[4.0s] Topk-2 Ocean                          0.3082
#[2.0s]-[4.0s] Topk-3 Waves, surf                    0.2931
#
#[4.0s]-[6.0s] Topk-1 Stream                         0.8928
#[4.0s]-[6.0s] Topk-2 Trickle, dribble               0.1404
#[4.0s]-[6.0s] Topk-3 Boat, Water vehicle            0.1306
#
#[6.0s]-[8.0s] Topk-1 Stream                         0.8503
#[6.0s]-[8.0s] Topk-2 Trickle, dribble               0.1666
#[6.0s]-[8.0s] Topk-3 Raindrop                       0.1620
#
#[8.0s]-[10.0s] Topk-1 Stream                         0.8594
#[8.0s]-[10.0s] Topk-2 Raindrop                       0.4584
#[8.0s]-[10.0s] Topk-3 Rain                           0.1884
```


```bash
python3 inference.py samples/mg4kDY_hy6o.wav # Default is the SAT-T 2s model
```
Outputs:

```bash
#===== samples/mg4kDY_hy6o.wav =====
#[0.0s]-[2.0s] Topk-1 Stream                         0.9555
#[0.0s]-[2.0s] Topk-2 Trickle, dribble               0.5271
#[0.0s]-[2.0s] Topk-3 Rain                           0.3057
#
#[2.0s]-[4.0s] Topk-1 Stream                         0.8074
#[2.0s]-[4.0s] Topk-2 Trickle, dribble               0.6095
#[2.0s]-[4.0s] Topk-3 Water                          0.3593
#
#[4.0s]-[6.0s] Topk-1 Stream                         0.8058
#[4.0s]-[6.0s] Topk-2 Water                          0.3823
#[4.0s]-[6.0s] Topk-3 Waterfall                      0.3219
#
#[6.0s]-[8.0s] Topk-1 Stream                         0.8496
#[6.0s]-[8.0s] Topk-2 Water                          0.4074
#[6.0s]-[8.0s] Topk-3 Trickle, dribble               0.3548
#
#[8.0s]-[10.0s] Topk-1 Stream                         0.8025
#[8.0s]-[10.0s] Topk-2 Water                          0.4172
#[8.0s]-[10.0s] Topk-3 Trickle, dribble               0.3403
```

As we can see, SAT provides **high-confidence scores** over a **prolonged timeframe**.


As a counter-example, if one uses our ViT-B, we get:

```bash
python3 inference.py samples/mg4kDY_hy6o.wav -m audiotransformer_base 

# Prints:
#===== samples/mg4kDY_hy6o.wav =====
#[0.0s]-[2.0s] Topk-1 Trickle, dribble               0.6101
#[0.0s]-[2.0s] Topk-2 Stream                         0.4968
#[0.0s]-[2.0s] Topk-3 Water                          0.2289
#
#[2.0s]-[4.0s] Topk-1 Stream                         0.3074
#[2.0s]-[4.0s] Topk-2 Water                          0.2828
#[2.0s]-[4.0s] Topk-3 Trickle, dribble               0.2814
#
#[4.0s]-[6.0s] Topk-1 Stream                         0.7029
#[4.0s]-[6.0s] Topk-2 Trickle, dribble               0.2179
#[4.0s]-[6.0s] Topk-3 Waterfall                      0.1804
#
#[6.0s]-[8.0s] Topk-1 Stream                         0.5569
#[6.0s]-[8.0s] Topk-2 Waterfall                      0.1705
#[6.0s]-[8.0s] Topk-3 Trickle, dribble               0.1406
#
#[8.0s]-[10.0s] Topk-1 Stream                         0.2891
#[8.0s]-[10.0s] Topk-2 Waterfall                      0.1813
#[8.0s]-[10.0s] Topk-3 Water                          0.0930
```


One can change the models (`audiotransformer_tiny`, `SAT_T_1s`, `audiotransformer_small`,`SAT_S_2s`,`SAT_S_1s`, `audiotransformer_base`, `SAT_B_2s`, `SAT_B_1s` ):

```bash
python3 inference.py -m SAT_T_1s -c 1.0 samples/jkLRith2wcc.wav


### Prints:
#===== samples/jkLRith2wcc.wav =====
#[0.0s]-[1.0s] Topk-1 Silence                        0.2903
#[0.0s]-[1.0s] Topk-2 Vehicle                        0.2587
#[0.0s]-[1.0s] Topk-3 Boat, Water vehicle            0.0793

#[1.0s]-[2.0s] Topk-1 Stream                         0.8000
#[1.0s]-[2.0s] Topk-2 Raindrop                       0.1655
#[1.0s]-[2.0s] Topk-3 Rain                           0.1522

#[2.0s]-[3.0s] Topk-1 Stream                         0.8642
#[2.0s]-[3.0s] Topk-2 Raindrop                       0.1193
#[2.0s]-[3.0s] Topk-3 Trickle, dribble               0.1065

#[3.0s]-[4.0s] Topk-1 Stream                         0.7319
#[3.0s]-[4.0s] Topk-2 Raindrop                       0.1958
#[3.0s]-[4.0s] Topk-3 Rain                           0.1204
```

### Very Short delay inference

Chunk-size (delay) can be controlled via `-c duration`.
For example, in the extreme case that our ViT-B (mAP 47.40) model only processes a single patch (**160 ms**) for the above samples, we get:

```bash
python3 inference.py samples/mg4kDY_hy6o.wav -m audiotransformer_base -c 0.16

#===== samples/mg4kDY_hy6o.wav ===== 
#[0.0s]-[0.16s] Topk-1 Music                          0.2091                  
#[0.0s]-[0.16s] Topk-2 Whoosh, swoosh, swish          0.1027
#[0.0s]-[0.16s] Topk-3 Silence                        0.0909 
#                                                                                                
#[0.16s]-[0.32s] Topk-1 Static                         0.1030
#[0.16s]-[0.32s] Topk-2 Silence                        0.1015
#[0.16s]-[0.32s] Topk-3 Single-lens reflex camera      0.0815
#
#[0.32s]-[0.48s] Topk-1 Static                         0.0839
#[0.32s]-[0.48s] Topk-2 Clatter                        0.0661
#[0.32s]-[0.48s] Topk-3 Cacophony                      0.0509
#
#[0.48s]-[0.64s] Topk-1 Electric toothbrush            0.0995
#[0.48s]-[0.64s] Topk-2 Clatter                        0.0834
#[0.48s]-[0.64s] Topk-3 Static                         0.0701
#
#[0.64s]-[0.8s] Topk-1 Electric toothbrush            0.1715
#[0.64s]-[0.8s] Topk-2 Static                         0.0629
#[0.64s]-[0.8s] Topk-3 Idling                         0.0511
#
#[0.8s]-[0.96s] Topk-1 Static                         0.1471
#[0.8s]-[0.96s] Topk-2 Cacophony                      0.1242
#[0.8s]-[0.96s] Topk-3 Electric toothbrush            0.0929
```

The above showed that our best model (but also all other SOTAs) are completely unable to provide *reasonable* results and cannot *differentiate between water and white noise*.

However, if we use `SAT_T_1s`:

```bash
python3 inference.py samples/mg4kDY_hy6o.wav -m SAT_T_1s -c 0.16

#===== samples/mg4kDY_hy6o.wav =====
#[0.0s]-[0.16s] Topk-1 Whoosh, swoosh, swish          0.6725
#[0.0s]-[0.16s] Topk-2 Sound effect                   0.1116
#[0.0s]-[0.16s] Topk-3 Music                          0.1057
#
#[0.16s]-[0.32s] Topk-1 Silence                        0.3615
#[0.16s]-[0.32s] Topk-2 Whoosh, swoosh, swish          0.1162
#[0.16s]-[0.32s] Topk-3 Static                         0.0956
#
#[0.32s]-[0.48s] Topk-1 Rain on surface                0.5616
#[0.32s]-[0.48s] Topk-2 Rain                           0.3506
#[0.32s]-[0.48s] Topk-3 Raindrop                       0.3257
#
#[0.48s]-[0.64s] Topk-1 Rain on surface                0.4327
#[0.48s]-[0.64s] Topk-2 Rain                           0.2457
#[0.48s]-[0.64s] Topk-3 Silence                        0.2116
#
#[0.64s]-[0.8s] Topk-1 Rain on surface                0.6230
#[0.64s]-[0.8s] Topk-2 Rain                           0.4475
#[0.64s]-[0.8s] Topk-3 Raindrop                       0.3959
#
#[0.8s]-[0.96s] Topk-1 Rain on surface                0.3707
#[0.8s]-[0.96s] Topk-2 Rain                           0.2532
#[0.8s]-[0.96s] Topk-3 Raindrop                       0.2089

```

While the results are not perfect, it is clear that SAT is far more superior for short-delay inference.


## Dataset acquisition

We propose simple preprocessing scripts in `datasets/` for `Audioset`.

For getting the (balanced) Audioset data you need [gnu-parallel](https://www.gnu.org/s/parallel) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).
Downloading the balanced Audioset is quick ( ~ 30 minutes):

```bash
cd datasets/audioset/
./1_download_audioset.sh
# After having downloaded the dataset, dump the .wav to .h5
./2_prepare_data.sh
```

Additionally, the `1_download_audioset.sh` script downloads our `PSL-labels` used in this work for the `balanced` and `full` training-subsets.


## Training


### MAE

After having prepared the data, to pretrain using MAE:

```bash
python3 1_run_mae.py config/mae/mae_tiny.yaml # Or the other configs
```

### SAT

Training a SAT is also simple:

```bash
python3 2_train_sat.py config/sat/balanced_sat_2_2s.yaml
```

After having trained the model you can use it for inference as:

```bash
python3 inference.py -m $PATH_TO_YOUR_CHECKPOINT samples/jkLRith2wcc.wav
```

