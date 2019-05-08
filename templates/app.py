# web-app for API image manipulation
from __future__ import print_function
from flask import Flask, request, render_template, send_from_directory, send_file
import os
from PIL import Image
import csv
import pytesseract
import pandas as pd
import numpy as np
import requests
import json
from io import BytesIO
import sqlalchemy
import datetime
import cv2



app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".jpeg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)


    #Homography
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.9

    def alignImages(im1, im2):

      # Convert images to grayscale
      im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

      # Detect ORB features and compute descriptors.
      orb = cv2.ORB_create(MAX_FEATURES)
      keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
      keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

      # Match features.
      matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
      matches = matcher.match(descriptors1, descriptors2, None)

      # Sort matches by score
      matches.sort(key=lambda x: x.distance, reverse=False)

      # Remove not so good matches
      numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
      matches = matches[:numGoodMatches]

      # Draw top matches
      imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
      cv2.imwrite("matches.jpg", imMatches)

      # Extract location of good matches
      points1 = np.zeros((len(matches), 2), dtype=np.float32)
      points2 = np.zeros((len(matches), 2), dtype=np.float32)

      for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

      # Find homography
      h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

      # Use homography
      height, width, channels = im2.shape
      im1Reg = cv2.warpPerspective(im1, h, (width, height))

      return im1Reg, h

    if __name__ == '__main__':

      # Read reference image
      refFilename = "/".join([target, "form.jpg"])
      print("Reading reference image : ", refFilename)
      imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

      # Read image to be aligned
      print("Reading image to align : ", destination);
      im = cv2.imread(destination, cv2.IMREAD_COLOR)

      print("Aligning images ...")
      # Registered image will be resotred in imReg.
      # The estimated homography will be stored in h.
      imReg, h = alignImages(im, imReference)

      # Write aligned image to disk.
      outFilename = destination
      print("Saving aligned image : ", outFilename);
      cv2.imwrite(outFilename, imReg)

      # Print estimated homography
      print("Estimated homography : \n",  h)



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

      #setting
      title = dfm.loc[(dfm.x > 230) & (dfm.x <520) & (dfm.y > 70) & (dfm.y < 110)]
      fullname = dfm.loc[(dfm.x > 230) & (dfm.x <520) & (dfm.y > 110) & (dfm.y < 170)]
      cardid = dfm.loc[(dfm.x > 230) & (dfm.x <520) & (dfm.y > 165) & (dfm.y < 200)]
      edate = dfm.loc[(dfm.x > 230) & (dfm.x <520) & (dfm.y > 210) & (dfm.y < 250)]
      #print(billno['word'])
      hel=' '.join(title.word)
      hel2=' '.join(fullname.word)
      hel3=' '.join(cardid.word)
      hel4=' '.join(edate.word)
      print(hel)
      print(hel2)
      print(hel3)
      print(hel4)


    # forward to processing page
    return render_template("processing.html", image_name=filename, hel=hel, hel2=hel2, hel3=hel3, hel4=hel4)



# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')







































## pdf
#@app.route("/PDF", methods=["POST"])
#def PDF():
#    # retrieve parameters from html form
#    filename = request.form['image']
#
#    # open and process image
#    target = os.path.join(APP_ROOT, 'static/images')
#    destination = "/".join([target, filename])
#
#    pdf = pytesseract.image_to_pdf_or_hocr(destination, extension='pdf')
#    destination2 = "/".join([target, "output.pdf"])
#    f = open(destination2, "w+b")
#    f.write(bytearray(pdf))
#    f.close()
#
#    return send_file(destination2,
#                     mimetype='application/pdf',
#                     attachment_filename='output.pdf',
#                     as_attachment=True)
