# ライブラリのインポート
import sys
import os
from flask import Flask, send_file, jsonify,render_template,redirect,url_for,request
import shutil
import random
import zipfile
import colorama
from colorama import Fore, Back, Style
app = Flask(__name__)

# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#! 自作関数はそれぞれfunctionフォルダに格納しています
from function import camera  # カメラを使う
from function import extraction_fish  # さかなの輪郭を使った処理(抽出・部位推定)
from function import createobj as obj  # 3DOBJを作る
from function import make_texture as texture # UV関係を作る
from function import settimer  # 時間測定
from function import painttool as paint  # ペイントツール
from function import module # 共通するモジュール

# from function import pre_createobj as obj

#* ======================================================================================== 
#* グローバル変数と魚の管理
#* ======================================================================================== 
g_fishNum = 1
g_templateNum = 1
g_upnum = 1
g_templateFlag = False

def addTemplateNum(addNum = 1):
    """テンプレートを加算する

    Args:
        addNum (int): 加算する数字. Defaults to 1.
    """
    global g_templateNum
    if g_templateNum <= 14:
        g_templateNum += addNum
    else:
        g_templateNum = 1

#* ======================================================================================== 
#* サーバー関係
#* ======================================================================================== 
#! 初期設定
def serverStart(ip = '150.89.239.35',flag=True):
    """サーバー起動関係

    Args:
        ip (str): サーバのIPアドレス. Defaultは '150.89.239.35'.
        flag : デバッグモードを使用するか判定するフラグ
    """
    # 全く意味がないアスキーアートを表示する
    print(Fore.LIGHTCYAN_EX + '               __   __')
    print(Fore.LIGHTCYAN_EX + '              __ \ / __')
    print(Fore.LIGHTCYAN_EX + '             /  \ | /  ')
    print(Fore.LIGHTCYAN_EX + '                 \|/')
    print(Fore.LIGHTCYAN_EX + '            _,.---v---._')
    print(Fore.LIGHTCYAN_EX + '   /\__/\  /            |')
    print(Fore.LIGHTCYAN_EX + '   \_  _/ /              |')
    print(Fore.LIGHTCYAN_EX + '     \ \_|           @ __|')
    print(Fore.LIGHTCYAN_EX + '  hjw \                \_')
    print(Fore.LIGHTCYAN_EX + '  `97  \     ,__/       /')
    print(Fore.LIGHTCYAN_EX + '     ~~~`~~~~~~~~~~~~~~/~~~~')

    print(Fore.LIGHTYELLOW_EX + '以下のURLにログインしてください')
    print(Fore.LIGHTYELLOW_EX + "-> http://127.0.0.1:5000")
    print(Fore.WHITE)
    print(Style.RESET_ALL+"",end="")
    
    app.run(host=ip, port=5000, debug=flag, use_reloader=False, threaded = False)

#! 最新の魚を返す
@app.route('/download/new', methods=['GET'])
def download_file_new():
    """最新のOBJと対応するPNGを送信する

    Returns:
        ZIPファイル
    """
    module.printTerminal("最新の魚を表示します")

    global g_upnum

    zipPath = f"./upload/{g_upnum-1}.zip"
    
    # ZIPファイルを送信
    return send_file(zipPath)

#! ランダムに生成された魚を返す
@app.route('/download/rand', methods=['GET'])
def download_file_rand():
    """ランダムなOBJと対応するPNGを送信する

    Returns:
        ZIPファイル
    """
    global g_upnum
    
    module.printTerminal("ランダムに魚を表示します")
    
    if g_upnum >= 2:
        num = random.randint(1,g_upnum-1)
    else:
        num = 1
        
    zipPath = f"./upload/{num}.zip"
    
    # ZIPファイルを送信
    return send_file(zipPath)

#! 魚を登録する
@app.route('/regit',methods=['GET'])
def regitFish():
    """魚の登録
    """
    global g_fishNum
    global g_templateNum
    global g_templateFlag
    
    module.printTerminal("魚を生成")
    
    file,g_templateFlag = module.regitFishPath(g_fishNum,g_templateNum)
    print(file)
    
    #? ------------------------------------------------------
    #? 画像から魚抽出->輪郭抽出
    #? ------------------------------------------------------
    module.printTerminal("画像から魚を抽出",2)
    preproTimer = settimer.timer("-->Preprocessing-Timer")

    filePath = "./inputimg/" + file
    img = extraction_fish.findFishParts(filePath, file)
    import cv2
    cv2.imwrite("./static/images/new.png", img)

    preproTimer.stop()

    #? ------------------------------------------------------
    #? 3DOBJに変換
    #? ------------------------------------------------------
    module.printTerminal("3DOBJを作成",2)
    
    createTimer = settimer.timer("-->create-Timer")

    # 拡張子を除くファイル名を作る
    file = os.path.splitext(file)[0]
    filePath = "./output/mask/" + file + "_mask3.png"
    maskPath = "./output/mask/" + file + "_mask3.png"
    obj.creating3D(filePath, maskPath, file, 100, False)
    createTimer.stop()
    
    # UVを貼る
    meshpath = "./output/mesh/me_" + file + ".obj"
    onlyFishImg = "./output/onlyfish/only" + file + ".png"
    texture.makeUVs(meshpath,onlyFishImg,meshpath,file)
    
    # flagがTrueならテンプレートを使用
    if g_templateFlag == True:
        # templateに加算する
        addTemplateNum()
    else:
        g_fishNum += 1
        
    # 魚をzip化する
    module.printTerminal("魚をzip化(送信用フォルダ化)",2)
    zipFish(meshpath,"./output/onlyfish/up2down_" + file + ".png")
    
    return download_file_new()

#! 魚をzip化する
def zipFish(fishPath,pngPath):
    """魚をZIP化します
    """
    global g_upnum
    
    # パスを取得
    zipPath = f"./upload/{g_upnum}.zip"
    g_upnum += 1
    
    # 情報をターミナルに出力
    print(fishPath)
    print(pngPath)
    
    # OBJまたはPNGがなければエラーを送信
    if not os.path.exists(fishPath) or not os.path.exists(pngPath):
        return jsonify({"error": "File not found"}), 404
    
    # ZIPファイルを作成
    with zipfile.ZipFile(zipPath, 'w') as zipf:
        zipf.write(fishPath, os.path.basename(fishPath))
        zipf.write(pngPath, os.path.basename(pngPath))

#! ペイントツールを起動させる(リフレッシュを兼ねてホームページを返す)
@app.route('/paint')
def showPaintTool():
    """ペイントツールを起動

    Returns:
        indexにリダイレクト
    """
    module.printTerminal("ペイントツールを起動します")
    paint.main()
    return redirect(url_for('index'))

#! カメラ起動
@app.route('/camera')
def useCamera():
    """カメラにて手描き写真を使用

    Returns:
        indexにリダイレクト
    """
    # カメラ起動->画像入手
    module.printTerminal("カメラにて撮影を行います")
    camera.camera_start()
    return redirect(url_for('index'))


#! 再起動
@app.route('/reset')
def resetApp():
    """システムを再起動
       ※プログラム内部の処理のみで，Flaskはそのまま

    Returns:
        indexにリダイレクト
    """
    global g_templateNum
    global g_fishNum
    global g_upnum
    
    module.printTerminal("システムを再起動します")
    
    # ディレクトリの削除
    EndOfProgram()
    
    # 初期設定
    module.initmain()
    g_fishNum = g_templateNum = g_upnum = 1
    
    return redirect(url_for('index'))

#! ホームページを表示
@app.route('/',methods=['GET'])
def index():
    """ホームページを表示

    Returns:
       html : Webサイト
    """
    module.printTerminal("HPを表示します")
    return render_template('index.html')


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """システムをシャットダウン

    Returns:
        str : メッセージ
    """
    module.printTerminal("シャットダウンします")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return redirect(url_for('index'))

#* ======================================================================================== 
#* main関数関係
#* ======================================================================================== 

def EndOfProgram():
    """プログラム終了時に実行する
    """
    if os.path.isdir('./output'):
        shutil.rmtree('./output')    
        
    if os.path.isdir('./upload'):
        shutil.rmtree('./upload')    
    
    # if os.path.isdir('./inputimg'):
    #     shutil.rmtree('./inputimg')
    
    # ペイントツールで作成したイラストを消す
    i = 1
    while True:
        filePath = './inputimg/fish_data('+ str(i) +').png' 
        if os.path.isfile(filePath):
            os.remove(filePath)
        else:
            break
        
        i += 1

    if os.path.isfile('./static/images/new.png' ):
        os.remove('./static/images/new.png' )
    
    if os.path.isfile('./static/images/paint.png' ):
        os.remove('./static/images/paint.png' )

def main():
    """すべての始まり"""
    debug = True

    if not debug:
        import logging
        log = logging.getLogger('werkzeug')
        log.disabled = True
    
    # ターミナルに色付き文字を出力するための設定
    colorama.init()
    
    # プログラムの開始を宣言
    module.printTerminal("Start Program")
    
    # 初期設定
    module.initmain()
    
    # サーバ起動
    serverStart('0.0.0.0',debug)
    
    # 削除関係
    if not debug:
        EndOfProgram()
    
    module.printTerminal("End Program")


if __name__ == "__main__":
    main()
