# Image manipulation API

from flask import Flask, render_template, send_from_directory, redirect, send_file
import os
from PIL import Image
from bs4 import BeautifulSoup
import csv
import pytesseract
import pandas as pd
import numpy as np

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

# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()
