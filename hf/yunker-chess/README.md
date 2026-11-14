---
license: cc-by-nc-4.0
pretty_name: YunkerChess
task_categories:
  - image-classification
tags:
  - chess
  - computer-vision
  - image-to-fen
  - youtube
size_categories:
  - 10K<n<100K
configs:
  - config_name: frames
    default: true
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
  - config_name: games
    data_files:
      - split: train
        path: games/train.jsonl
      - split: validation
        path: games/validation.jsonl
      - split: test
        path: games/test.jsonl
dataset_info:
  - config_name: games
    features:
      - name: id
        dtype: string
      - name: video_id
        dtype: string
      - name: youtube_url
        dtype: string
      - name: game
        dtype: int64
      - name: moves
        sequence: string
      - name: san
        sequence: string
      - name: move_frames
        sequence: int64
      - name: move_times
        sequence: float64
      - name: final_fen
        dtype: string
      - name: pawns_left
        list:
          - name: ply
            dtype: int64
          - name: swapped_at
            dtype: float64
---

# YunkerChess

Frames of over-the-board blitz games from the YouTube channel
[The Yunkers' Chess & Co](https://www.youtube.com/channel/UCwaKAQ6ZGQUC8_HhOZ20Vlw), each labelled with the
position on the board (as a FEN), and the games themselves, with when each move was played. The videos show a 2D
board beside the real one, and the labels are read from it.

**The dataset holds no images.** Each row names a YouTube video, a frame of it and the board region to crop. Make
the crops with [`crop_yunker.py`](crop_yunker.py), which downloads the videos:

```bash
uv run crop_yunker.py --out yunker_crops
# or, with Python 3.12+: pip install av==19.0.2 opencv-python-headless==4.11.0.86 "numpy>=2,<3" "yt-dlp>=2026.10.26"; python crop_yunker.py ...
```

`av` and OpenCV are pinned to the versions the dataset's `crop_hash` values were made with, as other versions can
decode or resize frames to slightly different pixels. The headless OpenCV build gives the same pixels as the full
one, and installs without GUI libraries.

It writes `yunker_crops/{train,validation,test}/<id>.jpg` with a `metadata.jsonl` beside them, so the crops load
with `datasets.load_dataset("imagefolder", data_dir="yunker_crops")`. It needs ffmpeg. Videos taken down from
YouTube since are skipped. Each crop is checked against the dataset's `crop_hash`, and a video whose crops don't
match (a download of another encoding, or frames out of step) is reported.

## Splits

| Split | Frames | Positions | Games | Videos |
| :--- | ---: | ---: | ---: | ---: |
| train | 62,321 | 21,627 | 533 | 502 |
| validation | 7,074 | 2,518 | 62 | 56 |
| test | 5,085 | 1,770 | 44 | 42 |

Splits are by whole video: no video is in two. Validation is 10% of the frames that aren't test, as whole videos
picked by a hash of their ids. The test videos are the ones the
[ChessQ ChessCam](https://huggingface.co/PBatch23888/chessq-chesscam) model's move-tracking evaluation replays,
held back from its training.

## Frames (`frames`, the default)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | `<video_id>_<frame>`, unique |
| `video_id` | string | The YouTube video id |
| `youtube_url` | string | `https://www.youtube.com/watch?v=<video_id>` |
| `game` | int | Which game of the video, from 1 (a few videos hold two) |
| `frame` | int | The frame's index in the video, from 0, decoding every frame in order |
| `time` | float | The frame's time in the video, in seconds (`frame / fps`) |
| `fen` | string | The piece placement, the first field of a FEN (`rnbqkbnr/pppppppp/8/...`), rank 8 first |
| `ply` | int | Moves played to reach the position (1 after White's first move) |
| `since_move` | float | Seconds since the board changed to this position |
| `until_move` | float or null | Seconds until the board next changes; null for a game's last position |
| `board_box` | int[4] | The board region to crop, `[x, y, width, height]` in the video's pixels, within the frame |
| `overlay_box` | int[4] | The 2D board's region, `[x, y, width, height]`. It shows the answer, so any part of it inside the crop is blacked out |
| `video_width`, `video_height` | int | The size of the video the boxes are for (mostly 1280 × 720) |
| `fps` | float | The video's frame rate |
| `model_agreement` | int or null | Squares (of 64) that a chess-reading model read as the label says; null for frames checked by an earlier model |
| `crop_hash` | string | A 64-bit difference hash of the crop (`dhash` in `crop_yunker.py`), as 16 hex digits |

The FEN is the board as it stands in the frame, with the board in whatever orientation the camera shows it. A
promotion where the player left the pawn on the square for a while is labelled as that pawn until it was swapped.

`since_move` and `until_move` are from when the board changed, which is about 0.3 s before the 2D board showed the
move (the median over the test games). A frame is never closer than 0.5 s after a move or 2 s before one.

`crop_hash` tells a crop made again from YouTube apart from the dataset's when the frame or region is wrong, not
when it is a frame or two off: the board barely changes between frames.

## Games (`games`)

One row per game that has frames in the split, for move tracking: following a game through its video and reading
off the moves.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | `<video_id>_game_<game>` |
| `video_id`, `youtube_url`, `game` | | As in `frames` |
| `moves` | string[] | The moves, in UCI (`e2e4`) |
| `san` | string[] | The same moves in SAN (`e4`) |
| `move_frames` | int[] | The frame of the video at which the 2D board showed each move |
| `move_times` | float[] | The same, in seconds |
| `final_fen` | string | The piece placement after the last move |
| `pawns_left` | object[] | Promotions where the pawn was left on the board: `ply`, and `swapped_at`, the second it was swapped for the new piece, or null if it stood until its square next changed |

The 2D board shows a move about 0.3 s after it is made on the board, and a move can show a little late when the
players move fast. In one test game a move was taken back, which the moves here don't show.

## How the frames were chosen

The moves and their times were read from the 2D board in each video. Frames are only taken while a position stood
still: none from 2 s before a move to 0.5 s after it, when a piece is in the air or in a hand. Each position gives
up to 3 frames, spread over the time it stood, and each game up to 150, so long games don't outweigh short ones.
Hands over the board are kept, as what they hide is still on it. The start position isn't used, as the videos open
with a title over the board.

Each frame's label was checked against a chess-reading model, trained on none of the test videos
(`model_agreement`): a frame it disagrees with on more than 16 squares was dropped, as that is a takeback, a camera
cut or the board being reset rather than a hard frame, and a game losing over 20% of its frames was left out.
Frames it read a few squares wrong are kept, as those are the hard ones. The test frames and 47,921 of the train
and validation frames were checked this way; the other 21,474 come from an earlier build that checked them with an
earlier model, and have no `model_agreement`.

The labels are automatic, so some are wrong: a move the 2D board showed late or early, or a square the board
region cuts off.

## License

The labels and boxes are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The
videos belong to The Yunkers' Chess & Co and are not part of the dataset; crops made from them are subject to
YouTube's terms and the channel's rights.
