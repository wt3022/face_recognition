from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import configparser
import json
import csv
import cv2

# 定数
END_KEY = 27                       # 終了するキーのキーコード(Ese)
SHUTTER_KEY = 13                   # シャッターキーのキーコード(Enter)
STOP_TIME = 20                     # 待ち時間
WINDOW_NAME2 = "shooting_mode"     # 表示ウィンドウの名前
BORDER_COLOR1 = (0, 0, 255)        # 枠線のカラーコード
BORDER_THICKNESS = 3               # 枠線の太さ

def shooting_mode():
    # configファイルの読み込み
    config_ini = configparser.ConfigParser()
    config_ini.read('./config.ini', encoding='utf-8')

    db_data = json.loads(config_ini['config'].get('datas'))
    AI_const = json.loads(config_ini['config'].get('const'))
    
    del config_ini
    
    # モデルとパラメータの読み込み 
    model = load_model('./model/use_model.hdf5')
    model.load_weights('./model/use_weight.hdf5')

    # 一度実行していないと重くなるため零行列で実行
    test = np.zeros((128, 128, 3)).reshape(-1, 128, 128, 3).astype('float32')
    test_re = model.predict(test,verbose=0)

    # メモリ解放
    del test
    del test_re

    # カメラの映像
    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    flag, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    
    # 撮影用の赤枠
    cv2.rectangle(frame, (148, 68), (492, 412), BORDER_COLOR1, thickness = BORDER_THICKNESS)

    img_array = np.array([])

    # カメラの動作フラグ
    run_flag = True

    # 撮影残り回数
    n = 3

    csv_in = []
    # ループ開始
    while flag and n > 0:
        if run_flag:
            cv2.putText(frame, str(n)+' remanining', (420,467), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),thickness=2)
            cv2.imshow(WINDOW_NAME2, frame)
            key = cv2.waitKey(20)

            # Enterで撮影
            if key == SHUTTER_KEY:
                # ボタンが押されたらフラグを折り、計算が終わるまで次の映像を流さないようにする
                run_flag = False
                img_array = img_to_array(cv2.resize(frame[148:492,68:412] , (128,128))).reshape(-1,128, 128, 3).astype('float32')
            
            # 終了するボタン、closeボタンが押された時の処理
            if key == END_KEY:
                break
            if cv2.getWindowProperty(WINDOW_NAME2, cv2.WND_PROP_VISIBLE) < 1:
                break

            flag, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (148, 68), (492, 412), BORDER_COLOR1, thickness = BORDER_THICKNESS)
        else:
            # AIの実行
            img_array /= 255
            feature_data =  model.predict(img_array,verbose=0)
            # 特徴データは2^16次元ベクトルにしておく
            feature_data = feature_data.reshape(65536).tolist()
            csv_in.append(feature_data)

            # 残り回数を減らし、フラグを戻す
            n = n - 1
            run_flag = True
    
    # 3枚ちゃんと撮影された際はconfigファイルに登録
    if len(csv_in) == 3:
        # csvが一つも登録されてない場合
        if len(db_data) == 0:
            # ハッシュ化
            hash = []
            for i,j in zip(AI_const[0],AI_const[1]):
                hash.append(sgn(csv_in[0][i] * j))
            
            # configに追加
            config = configparser.ConfigParser()
            config.add_section('config')
            config.set('config', 'const', json.dumps(AI_const))
            config.set('config', 'datas', json.dumps([["0.csv",hash]]))

            with open('config.ini', 'w') as file:
                config.write(file)
            
            # csvの作成
            with open('./data/0.csv','w') as file:
                writer = csv.writer(file)
                for i in range(3):
                    writer.writerow(csv_in[i])
        else:
            # ハッシュ化
            hash = []
            for i,j in zip(AI_const[0],AI_const[1]):
                hash.append(sgn(csv_in[0][i] * j))
            
            # ファイル名はn+1.csvとなる
            file_name = str(int((db_data[len(db_data) - 1][0]).replace('.csv',''))+1)+".csv"

            # configに追加
            db_data.append([file_name,hash])
            config = configparser.ConfigParser()
            config.add_section('config')
            config.set('config', 'const', json.dumps(AI_const))
            config.set('config', 'datas', json.dumps(db_data))

            with open('config.ini', 'w') as file:
                config.write(file)
            
            # csvの作成
            with open('./data/'+file_name,'w') as file:
                writer = csv.writer(file)
                for i in range(3):
                    writer.writerow(csv_in[i])

    cv2.destroyAllWindows()
    video_capture.release() 

def sgn(p):
    if p > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    shooting_mode()