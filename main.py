from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from multiprocessing import Process, Value, Array
import numpy as np
import configparser
import json
import ctypes
import csv
import cv2

# 定数
END_KEY = 27                       # 終了するキーのキーコード(Ese)
SHUTTER_KEY = 13                   # シャッターキーのキーコード(Enter)
STOP_TIME = 20                     # 待ち時間
WINDOW_NAME1 = "face_recognition"  # 表示ウィンドウの名前
BORDER_COLOR1 = (0, 0, 255)        # 通常時の枠線のカラーコード
BORDER_COLOR2 = (0, 255, 0)        # 承認時の枠線のカラーコード
BORDER_COLOR3 = (255, 0, 0)        # 承認拒否時の枠線のカラーコード
BORDER_THICKNESS = 3               # 枠線の太さ


# メイン関数
def main(AI_flag, AI_Value, AI_result):
    # HaarCascadeを読み込む
    cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt2.xml")

    # カメラの映像
    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # カメラの１フレームを取得
    flag, frame = video_capture.read()

    # カメラ映像の反転
    frame = cv2.flip(frame, 1)

    # 顔認証後のタイムラグ用変数
    stop_time = 0

    # ループ開始
    while flag:
        if stop_time == 0:

            # AIから返却された結果を確認する
            if AI_flag.value == 3:
                #print('AI_return:',AI_result.value)
                
                # 30フレームのラグを設ける 
                stop_time = 30

            # カメラ映像のGRAY化
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 顔候補のx,y座標と横幅、縦幅を取得化、枠線で囲う
            face_list = cascade.detectMultiScale(img_gray, minSize=(180, 180))
            
            for (x, y, w, h) in face_list:
                cv2.rectangle(frame, (x,y), (x+w, y+h), BORDER_COLOR1, thickness = BORDER_THICKNESS)
                in_img = frame[x:x+w,y:y+h]
                
            # 顔候補が一つ以上、フラグが1以上ならAIを並列で実行
            if len(face_list) != 0 and AI_flag.value == 2:
                AI_in_array = img_to_array(cv2.resize(in_img , (128,128))).reshape(-1, 128, 128, 3).astype('float32')
                AI_in_array /= 255
                temp = AI_in_array.reshape(49152).tolist()
                AI_Value[:] = temp
                AI_flag.value = 1
            

            
            # 映像（フレーム）を表示
            cv2.imshow(WINDOW_NAME1,frame)
            
            # キー入力待ち
            key = cv2.waitKey(STOP_TIME)
            
            # 終了するボタン、closeボタンが押された時の処理
            if key == END_KEY:
                break
            
            if cv2.getWindowProperty(WINDOW_NAME1, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 次のフレームを読み込む
            flag, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
        else:
            stop_time = stop_time - 1

            # 停止時間が戻ったら再度フラグを実行可能状態に戻す
            if stop_time == 0:
                AI_flag.value = 2

            # カメラ映像のGRAY化
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 顔候補のx,y座標と横幅、縦幅を取得化、枠線で囲う
            face_list = cascade.detectMultiScale(img_gray, minSize=(180, 180))
            
            # 顔認証の結果によって枠の色を変える
            if AI_result.value == 1:
                for (x, y, w, h) in face_list:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), BORDER_COLOR2, thickness = BORDER_THICKNESS)
            else:
                for (x, y, w, h) in face_list:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), BORDER_COLOR3, thickness = BORDER_THICKNESS)
            
            
            # 映像（フレーム）を表示
            cv2.imshow(WINDOW_NAME1,frame)
            
            # キー入力待ち
            key = cv2.waitKey(STOP_TIME)
            
            # 終了するボタン、closeボタンが押された時の処理
            if key == END_KEY:
                break
            
            if cv2.getWindowProperty(WINDOW_NAME1, cv2.WND_PROP_VISIBLE) < 1:
                break

            flag, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
    
    AI_flag.value = 4
    # breakしたらすべてのウィンドウを閉じる
    cv2.destroyAllWindows()
    # カメラを終了
    video_capture.release()   

def AI_run(AI_flag, AI_Value, AI_result):
    # モデルとパラメータの読み込み
    model = load_model('./model/use_model.hdf5')
    model.load_weights('./model/use_weight.hdf5')

    # コンフィグファイルを読み込む
    config_ini = configparser.ConfigParser()
    config_ini.read('./config.ini', encoding='utf-8')
    
    db_data = json.loads(config_ini['config'].get('datas'))
    AI_const = json.loads(config_ini['config'].get('const'))

    del config_ini

    # 待機用ループ開始
    while True:
        # フラグが1になったら実行
        if AI_flag.value == 1:
            AI_flag.value = 0
            # 配列を変換してAIに入力可能な形にする
            AI_in_array = np.array(AI_Value).reshape(-1, 128, 128, 3).astype('float32')
            
            # AIに入力結果を2^16次元ベクトルに変換
            feature_data = model.predict(AI_in_array,verbose=0)
            feature_data = feature_data.reshape(65536).tolist()

            # 射影用のパラメータを使用してハッシュ化
            feature_hash = []
            for i,j in zip(AI_const[0],AI_const[1]):
                feature_hash.append(sgn(feature_data[i] * j))
            
            # configに登録されているハッシュとハフマン距離が閾値以内のものを候補とする
            index_num = []
            if len(db_data) == 0:
                pass
            else:
                for i in range(len(db_data)):
                    print(9,np.sum(np.array(feature_hash) ^ np.array(db_data[i][1])))
                    if np.sum(np.array(feature_hash) ^ np.array(db_data[i][1])) < 500:
                        index_num.append(i)

            print('candidacy:',index_num)
            
            # 候補のCSVデータを読み込み
            if len(index_num) != 0:
                csv_datas = []
                for i in index_num:
                    with open('./data/'+db_data[i][0], 'r') as file:
                        read = csv.reader(file)
                        temp = []
                        for row in read:
                            if len(row) == 0:
                                pass
                            else:
                                temp.append(row)
                        csv_datas.append([db_data[i][0],temp])

                # 候補全てのコサイン類似度を測り、候補ごとの最大値を記憶
                max = []
                max_name = []
                for i in range(len(csv_datas)):
                    temp = np.array([])
                    for j in range(len(csv_datas[i][1])):
                        x = np.array(feature_data).reshape(65536).astype('float32')
                        y = np.array(csv_datas[i][1][j]).reshape(65536).astype('float32')
                        temp = np.append(temp, np.dot(x.T,y)/(np.linalg.norm(x) * np.linalg.norm(y)))
                    max.append(np.amax(temp))
                    max_name.append(csv_datas[i][0])

                print('Maxcos:',max)

                # max配列の中の最大値を最終候補とし、コサイン類似度が閾値を超えてたら認証
                if np.amax(np.array(max)) > 0.96:
                    print('candidate:',max_name[np.argmax(max)])
                    AI_result.value = 1
                else:
                    AI_result.value = 0
            else:
                AI_result.value = 0
            
            # フラグを3にし、mainが結果を確認できるようにする
            AI_flag.value = 3

        # 終了用フラグを検出したらbreak
        if AI_flag.value == 4:
            break

def sgn(p):
    if p > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    # プロセス間の共有変数
    AI_flag = Value('i',2)
    AI_result = Value('i',0)
    AI_Value = Array(ctypes.c_float,49152)

    '''
    フラグ設定
    AI_flag.value = 0  AI実行中
    AI_flag.value = 1  AI実行開始
    AI_flag.value = 2  AI実行可能
    AI_flag.value = 3  AI結果確認
    AI_flag.value = 4  プロセス終了
    '''

    # プロセス実行
    p2 = Process(target=AI_run, args=(AI_flag, AI_Value, AI_result,))
    p1 = Process(target=main, args=(AI_flag, AI_Value, AI_result,))
    p2.start()
    p1.start()

    # mainが終了したのにフラグが4にならないことがあるため再度フラグを4にする
    p1.join()
    AI_flag.value = 4
    p2.join()
