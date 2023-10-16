# 同じ顔の人を探すプログラム
[dlibライブラリ](http://dlib.net/)のCNNベースの顔抽出、顔比較機能を使って、同じ顔の人を探す。

## 0. 準備
```
# 依存ライブラリをインストールする
$ pip3 install -r requirements.txt

# 顔検出の機械学習モデルをダウンロード (get_faces.pyで使用する)
$ curl -O http://dlib.net/files/mmod_human_face_detector.dat.bz2
$ bunzip2 mmod_human_face_detector.dat.bz2

# 顔の特徴抽出の機械学習モデルをダウンロード (image_to_vector.pyで使用する)
$ curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 顔認識の機械学習モデルをダウンロード (image_to_vector.pyで使用する)
$ curl -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
$ bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
```

モデルデータのURLは、 <https://dlib.net/python/> を参照


## 1. 画像を縮小させる (大きいと処理時間が膨大になるので)
準備として、カレントディレクトリ以下に`image_original`ディレクトリを作って、そこに画像を置くこと。

```
$ python3 resize_image.py
```

このプログラムは`image_small`ディレクトリに縮小された画像を生成する。

TODO : いま0.2倍になっているので、元画像の大きさがいくつかに合わせて縮小倍率を変える


## 2. 画像から顔を抽出する
```
$ python3 get_faces.py
```

このプログラムは`image_small`ディレクトリの画像から抽出された顔を、96x96ピクセルのJPEG画像として`faces`ディレクトリに生成する。

ファイル名は、以下の形式：

```
[元のファイル名 (拡張子なし)]/[顔番号3桁ゼロ埋め0開始].jpg
```

## 3. 顔画像をベクトルに変換する
```
$ python3 image_to_vector.py
```

このプログラムは`faces`ディレクトリにある全ての顔が128次元ベクトルに変換し、`.npy`形式 (NumPyバイナリ) で`vecs`ディレクトリに生成する。


## 4. 顔のマッチングを行う
```
$ python3 match_faces.py
```

このプログラムは`vecs`ディレクトリの顔を総当りして同じ顔を探す。抽出された同じ顔のリストは、カレントディレクトリの`face_list.txt`にCSV形式で1行ごとに同じ顔のファイル名が列挙される。

このプログラムは乱数シードを固定しているため、プログラムを変更しない限りにおいて、何度実行しても同じ結果となる。


## 5. 同じ顔の視覚化
```
$ python3 visualize_face_list.py
```

このプログラムは`face_list.txt`をもとに、同じ顔の人物を最大6x6枚並べた画像を生成する。

