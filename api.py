# Image manipulation API

from flask import Flask, render_template, send_from_directory, redirect, send_file, request
import os
from PIL import Image
from bs4 import BeautifulSoup
import csv
import pytesseract
import pandas as pd
import numpy as np
import requests
import json
import sqlalchemy
import datetime
import pyodbc

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access redirects to API documentation
@app.route("/")
def main():
    return redirect("https://github.com/nicolasraj", code=302)
#######################################################################################################

@app.route("/ocr/<filename>", methods=["GET"])
def ocr(filename):

    # open and parse html
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    hocr = pytesseract.image_to_pdf_or_hocr(destination, extension='hocr')
    f = open("output.hocr", "w+b")
    f.write(bytearray(hocr))
    f.close()

    with open("output.hocr", "r") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')

        destination = "/".join([target, 'test.csv'])
        with open(destination, mode='w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(['Line','Word','Bbox','String'])
            for tag in soup.find_all("span"):
                if tag.get('class')==['ocr_line']:
                    lineno = tag.get('id')
                else:
                    spamwriter.writerow([lineno,tag.get('id'), tag.get('title'), tag.text])


    df = pd.read_csv(destination)

    #cleaning text
    df['Bbox']=df['Bbox'].str.replace('; x_wconf','')
    df['Bbox']=df['Bbox'].str.replace('bbox ','')

    #splitting the boundary box and word confidence
    df2= pd.DataFrame(df.Bbox.str.split(' ').tolist(),columns = ['x','y','x1','y1','conf'])


    #combining both the DataFrame
    dfm=pd.concat([df, df2], axis=1)
    dfm.x = dfm.x.astype(float)

    dfm['Column'] = np.where(dfm.x < 130, '1',
                   np.where(dfm.x < 560, '2',
                   np.where(dfm.x < 740, '3',
                   np.where(dfm.x < 900, '4','x'))))


    #dfm = dfm.drop(columns="Bbox")
    dfout = dfm[['Line', 'Word', 'Column', 'x', 'x1','y','y1','conf','String']]

    destination = "/".join([target, 'cleaned.csv'])
    dfout.to_csv(destination, index=None)


    return send_file(destination,
                     mimetype='text/csv',
                     attachment_filename='cleaned.csv',
                     as_attachment=True)

@app.route("/PDF/<filename>", methods=["GET"])
def PDF(filename):
    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    pdf = pytesseract.image_to_pdf_or_hocr(destination, extension='pdf')
    destination2 = "/".join([target, "output.pdf"])
    f = open(destination2, "w+b")
    f.write(bytearray(pdf))
    f.close()

    return send_file(destination2,
                     mimetype='application/pdf',
                     attachment_filename='output.pdf',
                     as_attachment=True)
#######################################################################################################


@app.route("/loan/<filename>", methods=["GET"])
def loan(filename):


    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    #from io import BytesIO
    # Replace <Subscription Key> with your valid subscription key.
    subscription_key = "35eb2945fec54af593e7b4770d2e1a3f"
    assert subscription_key

    vision_base_url = "https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/"

    ocr_url = vision_base_url + "ocr"

    # Set image_url to the URL of an image that you want to analyze.
    #image_url = "http://40.90.188.245:5000/static/images/output1.jpg"
    image_path = destination
    image_data = open(image_path, "rb").read()

    headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                  'Content-Type': 'application/octet-stream'}
    params  = {'language': 'unk', 'detectOrientation': 'true'}
    #data    = {'url': image_url}

    response = requests.post(
        ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()

    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    with open('static/images/csvfile.csv','w',encoding='utf-8') as file:
        file.write('x,y,x1,y1,word')
        file.write('\n')
        for line in line_infos:
            for word_metadata in line:
                for word_info in word_metadata["words"]:
                    word_infos.append(word_info)
                    str1=json.dumps(word_info)
                    data = json.loads(str1)
                    file.write(data['boundingBox']+',')
                    text = "\"" + data['text'] + "\""
                    file.write(text)
                    file.write('\n')



    dfm = pd.read_csv('static/images/csvfile.csv')
    dfm.x = dfm.x.astype(int)
    dfm.y = dfm.y.astype(int)

    ytop=0
    ybottom=100000

    dfm['Column'] = np.where((dfm.x < 185) & (dfm.y > ytop) & (dfm.y < ybottom) , '1',
                   np.where((dfm.x < 1300) & (dfm.y > ytop) & (dfm.y < ybottom), '2',
                   np.where((dfm.x < 1655) & (dfm.y > ytop) & (dfm.y < ybottom), '3',
                   np.where((dfm.x < 2100) & (dfm.y > ytop) & (dfm.y < ybottom), '4','5'))))

    #print(dfm)
    #print(dfm.to_string())

    dfm=dfm.sort_values(by='y', axis=0, ascending=True)
    dfm=dfm.reset_index()

    dfm['Line']=""

    ledger1=dfm['y'][0]-10
    ledger2=dfm['y'][0]+10
    no = 1
    no1= str(no)

    #
    #
    endpoint=len(dfm.index)
    for i in range(0,endpoint):
        if ledger1<=dfm['y'][i]<=ledger2:
            dfm['Line'][i]=no1
        else:
            ledger1=dfm['y'][i]-10
            ledger2=dfm['y'][i]+10
            no=no+1
            no1=str(no)
            dfm['Line'][i]=no1

    currentDT = datetime.datetime.now()
    dt=(str(currentDT))

    print(dfm.to_string())
    destination = "/".join([target, 'loan.csv'])
    dfm.to_csv(destination, index=None)
    engine = sqlalchemy.create_engine("mssql+pyodbc://nicolas:nasigorengikanmasin@pdserverip.ddns.net/NC_Testing?driver=SQL+Server+Native+Client+11.0")
    dfm.to_sql(dt+" Loan", engine , index=False, if_exists='append')
    print("Uploaded to DB")

    return send_file(destination,
                     mimetype='text/csv',
                     attachment_filename='loan.csv',
                     as_attachment=True)


#############################################################################################

@app.route("/loan2")
def loan2():


    # open and process image
    url = "https://www.wada-ama.org/sites/default/files/resources/thumbnails/tdssa_2017_eng_page_01.jpg"
    image_url = url

    #from io import BytesIO
    # Replace <Subscription Key> with your valid subscription key.
    subscription_key = "35eb2945fec54af593e7b4770d2e1a3f"
    assert subscription_key

    vision_base_url = "https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/"

    ocr_url = vision_base_url + "ocr"

    # Set image_url to the URL of an image that you want to analyze.
    #image_url = "http://40.90.188.245:5000/static/images/output1.jpg"
    #image_path = destination
    #image_data = open(image_path, "rb").read()

    headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                  'Content-Type': 'application/octet-stream'}
    params  = {'language': 'unk', 'detectOrientation': 'true'}
    data    = {'url': image_url}

    response = requests.post(
        ocr_url, headers=headers, params=params, json=data)
    response.raise_for_status()

    analysis = response.json()

    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    with open('static/images/csvfile.csv','w',encoding='utf-8') as file:
        file.write('x,y,x1,y1,word')
        file.write('\n')
        for line in line_infos:
            for word_metadata in line:
                for word_info in word_metadata["words"]:
                    word_infos.append(word_info)
                    str1=json.dumps(word_info)
                    data = json.loads(str1)
                    file.write(data['boundingBox']+',')
                    text = "\"" + data['text'] + "\""
                    file.write(text)
                    file.write('\n')



    dfm = pd.read_csv('static/images/csvfile.csv')
    dfm.x = dfm.x.astype(int)
    dfm.y = dfm.y.astype(int)

    ytop=0
    ybottom=100000

    dfm['Column'] = np.where((dfm.x < 185) & (dfm.y > ytop) & (dfm.y < ybottom) , '1',
                   np.where((dfm.x < 1300) & (dfm.y > ytop) & (dfm.y < ybottom), '2',
                   np.where((dfm.x < 1655) & (dfm.y > ytop) & (dfm.y < ybottom), '3',
                   np.where((dfm.x < 2100) & (dfm.y > ytop) & (dfm.y < ybottom), '4','5'))))

    #print(dfm)
    #print(dfm.to_string())

    dfm=dfm.sort_values(by='y', axis=0, ascending=True)
    dfm=dfm.reset_index()

    dfm['Line']=""

    ledger1=dfm['y'][0]-10
    ledger2=dfm['y'][0]+10
    no = 1
    no1= str(no)

    #
    #
    endpoint=len(dfm.index)
    for i in range(0,endpoint):
        if ledger1<=dfm['y'][i]<=ledger2:
            dfm['Line'][i]=no1
        else:
            ledger1=dfm['y'][i]-10
            ledger2=dfm['y'][i]+10
            no=no+1
            no1=str(no)
            dfm['Line'][i]=no1

    currentDT = datetime.datetime.now()
    dt=(str(currentDT))

    print(dfm.to_string())
    destination = "/".join([target, 'loan.csv'])
    dfm.to_csv(destination, index=None)
    engine = sqlalchemy.create_engine("mssql+pyodbc://nicolas:nasigorengikanmasin@pdserverip.ddns.net/NC_Testing?driver=SQL+Server+Native+Client+11.0")
    dfm.to_sql(dt+" Loan", engine , index=False, if_exists='append')
    print("Uploaded to DB")

    return send_file(destination,
                     mimetype='text/csv',
                     attachment_filename='loan.csv',
                     as_attachment=True)



######################################################################################################
@app.route("/loan4", methods=["GET"])
def loan4():
    # open and process image
    #from io import BytesIO
    # Replace <Subscription Key> with your valid subscription key.
    subscription_key = "35eb2945fec54af593e7b4770d2e1a3f"
    assert subscription_key

    vision_base_url = "https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/"

    ocr_url = vision_base_url + "ocr"

    # Set image_url to the URL of an image that you want to analyze.
    refId1=request.args['refId']
    url1=request.args['url']

    refid = refId1
    image_url = url1


    filename = refid

    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    #Download image and save to local folder
    img_data = requests.get(image_url).content
    with open(destination+".jpg", 'wb') as handler:
        handler.write(img_data)

    image_path = destination+".jpg"
    image_data = open(image_path, "rb").read()

    headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
                  'Content-Type': 'application/octet-stream'}
    params  = {'language': 'unk', 'detectOrientation': 'true'}

    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()

    analysis = response.json()


    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    with open('static/images/csvfile.csv','w',encoding='utf-8') as file:
        file.write('x,y,x1,y1,word')
        file.write('\n')
        for line in line_infos:
            for word_metadata in line:
                for word_info in word_metadata["words"]:
                    word_infos.append(word_info)
                    str1=json.dumps(word_info)
                    data = json.loads(str1)
                    file.write(data['boundingBox']+',')
                    text = "\"" + data['text'] + "\""
                    file.write(text)
                    file.write('\n')



    dfm = pd.read_csv('static/images/csvfile.csv')
    dfm.x = dfm.x.astype(int)
    dfm.y = dfm.y.astype(int)

    ytop=0
    ybottom=100000

    dfm['Column'] = np.where((dfm.x < 185) & (dfm.y > ytop) & (dfm.y < ybottom) , '1',
                   np.where((dfm.x < 1300) & (dfm.y > ytop) & (dfm.y < ybottom), '2',
                   np.where((dfm.x < 1655) & (dfm.y > ytop) & (dfm.y < ybottom), '3',
                   np.where((dfm.x < 2100) & (dfm.y > ytop) & (dfm.y < ybottom), '4','5'))))

    #print(dfm)
    #print(dfm.to_string())

    dfm=dfm.sort_values(by='y', axis=0, ascending=True)
    dfm=dfm.reset_index()

    dfm['Line']=""

    ledger1=dfm['y'][0]-10
    ledger2=dfm['y'][0]+10
    no = 1
    no1= str(no)

    #
    #
    endpoint=len(dfm.index)
    for i in range(0,endpoint):
        if ledger1<=dfm['y'][i]<=ledger2:
            dfm['Line'][i]=no1
        else:
            ledger1=dfm['y'][i]-10
            ledger2=dfm['y'][i]+10
            no=no+1
            no1=str(no)
            dfm['Line'][i]=no1

    currentDT = datetime.datetime.now()
    dt=(str(currentDT))


    print(dfm.to_string())
    destination = "/".join([target, 'loan.csv'])
    dfm.to_csv(destination, index=None)
    engine = sqlalchemy.create_engine("mssql+pyodbc://nicolas:nasigorengikanmasin@pdserverip.ddns.net/NC_Testing?driver=SQL+Server+Native+Client+11.0")
    dfm.to_sql(refid+"_"+dt+" Loan", engine , index=False, if_exists='append')
    print("Uploaded to DB")

    ####################################################JOGET#############################################################
    conn = pyodbc.connect("DRIVER={MySQL ODBC 5.3 Unicode Driver}; SERVER=192.168.0.148; PORT=3306; DATABASE=jwdb; UID=jogetv6; PASSWORD=jogetv6P@ss;")
    cursor = conn.cursor()

    endpoint=len(dfm.index)
    for i in range(0,endpoint):
        one=str(dfm['index'][i])
        two=str(dfm['x'][i])
        three=str(dfm['y'][i])
        four=str(dfm['x1'][i])
        five=str(dfm['y1'][i])
        six=str(dfm['word'][i])
        seven=str(dfm['Column'][i])
        eight=str(dfm['Line'][i])
        unique=refid+"_"+str(i)
        string1 = 'INSERT INTO app_fd_tr_docData(id, c_index, c_x, c_y, c_x1, c_y1, c_word, c_column, c_line, c_parentId) VALUE("'+unique+'","'+one+'","'+two+'","'+three+'","'+four+'","'+five+'","'+six+'","'+seven+'","'+eight+'","'+refid+'");'

        cursor.execute(string1)
    conn.commit()


   ##############################################################################################################################
    conn = pyodbc.connect("DRIVER={MySQL ODBC 5.3 Unicode Driver}; SERVER=192.168.0.148; PORT=3306; DATABASE=jwdb; UID=jogetv6; PASSWORD=jogetv6P@ss;")
    cursor = conn.cursor()

    string1 = 'UPDATE app_fd_tr_doc SET c_status = "Complete" WHERE id ="'+refid+'";'


    print(string1)
    cursor.execute(string1)
    conn.commit()


    ######################################################################################################################

    return send_file(destination,
                     mimetype='text/csv',
                     attachment_filename='loan.csv',
                     as_attachment=True)


@app.route("/loan5", methods=["GET"])
def loan5():


    # open and process image
    #from io import BytesIO
    # Replace <Subscription Key> with your valid subscription key.
    target = os.path.join(APP_ROOT, 'static/images')
    subscription_key = "35eb2945fec54af593e7b4770d2e1a3f"
    assert subscription_key

    vision_base_url = "https://southeastasia.api.cognitive.microsoft.com/vision/v2.0/"

    ocr_url = vision_base_url + "ocr"

    # Set image_url to the URL of an image that you want to analyze.
    refId1=request.args['refId']
    url1=request.args['url']

    refid = refId1
    image_url = url1
    #"https://i.imgur.com/WZAwdin.jpg"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params  = {'language': 'unk', 'detectOrientation': 'true'}
    data    = {'url': image_url}
    response = requests.post(ocr_url, headers=headers, params=params, json=data)
    response.raise_for_status()

    analysis = response.json()


    # Extract the word bounding boxes and text.
    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    with open('static/images/csvfile.csv','w',encoding='utf-8') as file:
        file.write('x,y,x1,y1,word')
        file.write('\n')
        for line in line_infos:
            for word_metadata in line:
                for word_info in word_metadata["words"]:
                    word_infos.append(word_info)
                    str1=json.dumps(word_info)
                    data = json.loads(str1)
                    file.write(data['boundingBox']+',')
                    text = "\"" + data['text'] + "\""
                    file.write(text)
                    file.write('\n')



    dfm = pd.read_csv('static/images/csvfile.csv')
    dfm.x = dfm.x.astype(int)
    dfm.y = dfm.y.astype(int)

    ytop=0
    ybottom=100000

    dfm['Column'] = np.where((dfm.x < 185) & (dfm.y > ytop) & (dfm.y < ybottom) , '1',
                   np.where((dfm.x < 1300) & (dfm.y > ytop) & (dfm.y < ybottom), '2',
                   np.where((dfm.x < 1655) & (dfm.y > ytop) & (dfm.y < ybottom), '3',
                   np.where((dfm.x < 2100) & (dfm.y > ytop) & (dfm.y < ybottom), '4','5'))))

    #print(dfm)
    #print(dfm.to_string())

    dfm=dfm.sort_values(by='y', axis=0, ascending=True)
    dfm=dfm.reset_index()

    dfm['Line']=""

    ledger1=dfm['y'][0]-10
    ledger2=dfm['y'][0]+10
    no = 1
    no1= str(no)

    #
    #
    endpoint=len(dfm.index)
    for i in range(0,endpoint):
        if ledger1<=dfm['y'][i]<=ledger2:
            dfm['Line'][i]=no1
        else:
            ledger1=dfm['y'][i]-10
            ledger2=dfm['y'][i]+10
            no=no+1
            no1=str(no)
            dfm['Line'][i]=no1

    currentDT = datetime.datetime.now()
    dt=(str(currentDT))

    print(dfm.to_string())
    destination = "/".join([target, 'loan.csv'])
    dfm.to_csv(destination, index=None)
    engine = sqlalchemy.create_engine("mssql+pyodbc://nicolas:nasigorengikanmasin@pdserverip.ddns.net/NC_Testing?driver=SQL+Server+Native+Client+11.0")
    engine = sqlalchemy.create_engine("mssql+pyodbc://192.168.0.148/jwdb?driver=SQL+Server+Native+Client+11.0")
    dfm.to_sql(refid+"_"+dt+" Loan", engine , index=False, if_exists='append')
    print("Uploaded to DB")

    return send_file(destination,
                     mimetype='text/csv',
                     attachment_filename='loan.csv',
                     as_attachment=True)

# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
