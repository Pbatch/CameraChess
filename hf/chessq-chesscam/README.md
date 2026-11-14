---
license: other
license_name: polyform-noncommercial-license-1.0.0
license_link: LICENSE
base_model:
  - timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k
library_name: chessqueries
tags:
  - chess
  - computer-vision
  - image-to-fen
  - litert
  - safetensors
  - mobilenetv4
---

# ChessQ ChessCam

A ChessQueries model for reading chess positions from ChessCam footage, exported to LiteRT for the browser: a
MobileNetV4 Conv-M encoder with a ChessQueries decoder, at 320 × 320. It needs 2.3 GMAC a frame and runs in about
100 ms on a phone's CPU.

Built with [ChessQueries](https://github.com/JSeytre/chessqueries)
([paper](https://arxiv.org/abs/2608.30762)).

## Files

| File | Encoder | Input | Format |
| :--- | :--- | :--- | :--- |
| `chessq-chesscam-mnv4m-320-v1-fp16.tflite` | MobileNetV4 Conv-M | 320 × 320 | LiteRT, FP16 weights (23.9 MB) |
| `chessq-chesscam-mnv4m-320-v1-fp32.safetensors` | MobileNetV4 Conv-M | 320 × 320 | PyTorch, FP32 weights (47.2 MB) |

The `.tflite` file runs with [LiteRT.js](https://ai.google.dev/edge/litert/web) on WebGPU or on the CPU
(WebAssembly). The model only works well at 320 × 320.

The `.safetensors` file is in the ChessQueries checkpoint format, but ChessQueries itself has no CNN encoders, so
it cannot build this model as it is. The encoder is timm's `mobilenetv4_conv_medium` (`features_only`), whose
stride-16 and stride-32 feature maps are each projected to 256 channels, given 2D sine position codes and a
per-level embedding, and passed to the ChessQueries decoder in place of a ViT's patch tokens.

## Datasets

The model is trained on every image of these datasets except YunkerChess's test split and ChessQ Test (112,785
images), with no other validation or test split held back. The splits shown are each dataset's own. YunkerChess's
test split (5,085 frames from 42 videos) is never trained on, as the model is tested on those videos instead (see
[Evaluation](#evaluation)), and ChessQ Test is a test set only:

| Dataset | Train | Val | Test | Total | Source |
| :--- | ---: | ---: | ---: | ---: | :--- |
| ChessCam | 15,590 | 71 | 2,275 | 17,936 | Frames from recorded [ChessCam](https://www.chesscam.net) games, labelled by converting the app's piece boxes and board grids to FENs |
| [YunkerChess](https://huggingface.co/datasets/PBatch23888/yunker-chess) | 62,321 | 7,074 | 5,085 | 74,480 | Frames from 600 blitz videos by [The Yunkers' Chess & Co](https://www.youtube.com/channel/UCwaKAQ6ZGQUC8_HhOZ20Vlw), labelled from the 2D board the videos show beside the real one, keeping only frames where the position had stood still for at least half a second |
| ChessReD | 6,479 | 2,192 | 2,129 | 10,800 | ChessReD |
| Synthetic | 4,527 | 140 | 333 | 5,000 | Blender renders |
| ChessCog | 4,400 | 146 | 342 | 4,888 | ChessCog |
| SLCC | 1,475 | 326 | 373 | 2,174 | SLCC |
| Digitizer | 1,027 | 141 | 188 | 1,356 | Webcam frames of 20 games from Oliver Butler's [Realtime-Chess-Digitizer](https://github.com/oliverbutler/Realtime-Chess-Digitizer) (CC0) |
| Samryan | 408 | 46 | 46 | 500 | Overhead photos from Sam Ryan and Mukund Vaidya's [chess-dataset](https://github.com/samryan18/chess-dataset) (MIT) |
| Fenit | 268 | 58 | 58 | 384 | Phone photos from "Annotated Image Dataset for Fenit: Automatic Chessboard Digitization and FEN Generation" (Harjanto, Oei and Susanto, Binus University; CC BY 4.0) |
| CVChess | 224 | 64 | 64 | 352 | CVChess, with the labels released with ChessQueries (CC BY-NC 4.0) |
| [ChessQ Test](https://huggingface.co/datasets/joelseytre/chessq-test) | – | – | 50 | 50 | Human-reviewed photos chosen as hard but fair cases, across 15 chess-set types, viewpoints, lighting and backgrounds, by Joël Seytre (CC BY-NC 4.0) |
| **All** | **96,719** | **10,258** | **10,943** | **117,920** | |

Every crop is exactly the board region, with no margin.

## Input and output

- **Input:** uint8 `(1, 320, 320, 4)`, RGBA pixels (for example, canvas `ImageData.data` as is).
  The image is resized to 320 × 320 without keeping its aspect ratio. Normalisation happens inside the model.
- **Outputs** (their order is up to the runtime, so tell them apart by shape):
  - float32 `(1, 64, 13)`: softmax probabilities for each square, with squares in FEN order (a8, b8, …, h1).
    The classes are `. P N B R Q K p n b r q k`, where `.` is an empty square.
  - float32 `(1, 64, 2)`: where each square's query attends in the input image, as `(x, y)` in [0, 1],
    squares in the same order. It is the attention-weighted centre of the image features in the last
    decoder layer's cross-attention, so it is pulled towards the board centre: use it for directions
    (for example, which way rank 8 lies), not exact square positions.

The probabilities are overconfident. Move tracking works best with them softened as softmax(log p / 3.0), with an
evidence weight of 2.5 (see [Evaluation](#evaluation)).

## Training

The encoder starts from timm's MobileNetV4 Conv-M weights (`mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k`,
ImageNet-12k then ImageNet-1k); the ChessQueries decoder (4 layers) starts from scratch. It was trained for 30
epochs at 320 × 320 on every image of the [datasets](#datasets) except YunkerChess's test split and ChessQ Test,
with batch 64, learning rate 3e-4 after 3 warm-up epochs, and bf16 mixed precision.

It was distilled from the earlier, larger ChessQ ChessCam v5 (11.3 GMAC a frame), which sees the same augmented
images: the loss is the cross-entropy with the labels plus the KL divergence from v5's
probabilities for each square, both at a temperature of 2 (scaled by its square). The same run trained on the
labels alone does worse in move tracking: 38 games with every move right, 92 corrections and 0.82 squares misread
per frame, against 39, 86 and 0.76 (see [Evaluation](#evaluation)).

## Evaluation

As every image of the other nine datasets was trained on, there is no held-out test set for them, and test-split
scores would not be meaningful.

What it is tested on instead is move tracking as in the [ChessCam web app](https://github.com/Pbatch/ChessCam),
replayed on recorded YunkerChess blitz games from 45 videos that are held back from training, against
[ChessQ Lite V4](https://huggingface.co/joelseytre/chessq-lite) (`cq-vit-s-644-v4-seed2`, at its 644 × 644 input).
The true moves are read from 42 of the videos (44 games), whose frames are the
[YunkerChess dataset](https://huggingface.co/datasets/PBatch23888/yunker-chess)'s test split, and the results are
for the 43 games (3,571 moves) in which no moves were taken back, each model at the tracker settings that suit it
best, at 4 frames a second:

| Metric | This model | ChessQ Lite V4 |
| :--- | ---: | ---: |
| Games with every move right | **39 (91%)** | 27 (63%) |
| Moves right before each game's first mistake | **3,415 (95.6%)** | 2,761 (77.3%) |
| Corrections needed to fix the moves found | **86 (2.0 per game)** | 405 (9.4 per game) |
| Squares misread per frame | **0.76** | 1.81 |

A correction is one move added, removed or changed (the edit distance from the true game); moves found after a
game's last move, as the board is cleared, are not counted. Squares misread per frame are counted over the 57,353
frames where the position stood still (at least 1.5 s from any move). The tracker settings are a temperature and an
evidence weight: 3.0 and 2.5 for this model (temperatures 2.5 to 4.0 did as well with weights about 0.83 × the
temperature, and none of a sweep from 2.0 to 4.0 did better), and for ChessQ Lite V4 3.5 and 1.05, the best of a
sweep from 2.5 to 4.5.

More than half of this model's misread squares are in one game (KxBHi8hM110), the hardest of the set; without it, it
misreads 0.34 squares a frame.

### ChessQ Test

[ChessQ Test](https://huggingface.co/datasets/joelseytre/chessq-test) scores single photos rather than move tracking:
50 hard photos, none trained on. Its maintainer ran each model on the whole photo, with no board cropping, whereas
this model is trained on crops of exactly the board region. An exact board is all 64 squares right:

| Model | Input | Exact boards | Square accuracy |
| :--- | ---: | ---: | ---: |
| This model (LiteRT, FP16 weights) | 320 × 320 | 7/50 (14%) | 89.41% |
| ChessQ ChessCam v5 | 320 × 320 | 10/50 (20%) | 90.62% |
| ChessQ Lite V5 | 644 × 644 | 18/50 (36%) | 96.72% |

On the 30 photos whose chess sets look like ChessQ's training examples, this model gets 1 board exact (85.21% of
squares); on the other 20, 6 (95.70%). Its card has the other models' results.

## Speed

Median time per run with LiteRT.js 2.6.0 in Chrome on a Pixel 7a, including reading the outputs back:

| Pixel 7a | This model (320 × 320) | ChessQ Lite V5 (644 × 644) | Speed-up |
| :--- | ---: | ---: | ---: |
| WebGPU (FP16) | 62 ms | 1,375 ms | 22× |
| CPU (WebAssembly, 8 threads) | 103 ms | 3,400 ms | 33× |

Times are from a cool phone. ChessQ Lite was exported to LiteRT the same way and timed earlier on the same phone,
as V4 (`cq-vit-s-644-v4-seed2`); V5 (`cq-vit-s-644-v5-seed1`) is the same ViT-S at the same input size, so it
runs in the same time. Under sustained load the phone throttles its CPU, and this model's CPU time grows to about
120 ms (ChessQ Lite's to about 4,500 ms); the WebGPU times barely change.

## License

The model is made with ChessQueries and distributed under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

Third-party components keep their upstream terms; see [the notices](NOTICE).

Required Notice: Copyright (c) 2026 Joël Seytre.
