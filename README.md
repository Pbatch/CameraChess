# Camera Chess

## TODO
* Label more data from outdoors
* Try beam search in tracker
* Fix data from "camera_chess/data_cleaning/sanity_check.py" script
* Make better validation set (include outdoor data)
* Benchmark TFLite model on mobile
* Fix override

## Links
* Chesscog - https://github.com/georg-wolflein/chesscog
* Play online - https://github.com/karayaman/Play-online-chess-with-real-chess-board
* ChessboardDetect - https://github.com/Elucidation/ChessboardDetect
* LiveChess2FEN - https://github.com/davidmallasen/LiveChess2FEN

## Papers
* Chesscog - https://github.com/georg-wolflein/chesscog-report/raw/master/report.pdf
* Chamfer matching - https://static1.squarespace.com/static/5fb9c8b175e654100b273fd3/t/5fd4b60d0f11661d4c015070/1607775761754/2018_ChessPieceRecognition_WACV.pdf

## Useful

Tar scripts for Google Drive
```bash
tar -zcvf models.tar.gz models
tar --exclude='*.webm' --exclude='*.tar.gz' --exclude='*.mp4' -zcvf data.tar.gz data
```
