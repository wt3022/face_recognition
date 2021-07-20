import os
import PySimpleGUI as sg
import configparser
import json




def deleat_data():
    # configの読み込み
    config_ini = configparser.ConfigParser()
    config_ini.read('./config.ini', encoding='utf-8')
    
    db_data = json.loads(config_ini['config'].get('datas'))
    AI_const = json.loads(config_ini['config'].get('const'))
    
    del config_ini
    
    # プルダウンリストの作成
    combo_data = []
    for i in range(len(db_data)):
        combo_data.append(db_data[i][0])
    
    # GUIの定義
    sg.theme('DarkTeal7')
    layout = [ [sg.Text('登録削除したいファイルを指定してください。')],
            [sg.Combo(combo_data, default_value="選択", size=(15,1), key='slect', readonly=True)],
            [sg.Button('キャンセル', key='cancel'),sg.Button('OK', key='ok')]
            ]
            
    window = sg.Window('登録削除', layout)
    
    while True:
        event, values = window.read()
        
        # Closeイベント
        if event == sg.WIN_CLOSED:
            break
        
        # キャンセルイベント
        if event == 'cancel':
            break
        
        # OKイベント
        if event == 'ok':
            # deleteファイル名を受け取り、削除
            for i in range(len(db_data)):
                if values['slect'] == db_data[i][0]:
                    os.remove('./data/'+db_data[i][0])
                    del db_data[i]
                    break
            
            # configの更新
            config = configparser.ConfigParser()
            config.add_section('config')
            config.set('config', 'const', json.dumps(AI_const))
            config.set('config', 'datas', json.dumps(db_data))
                    
            with open('config.ini', 'w') as file:
                config.write(file)
            break

    window.close()

if __name__ == '__main__':
    deleat_data()