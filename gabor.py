import cv2
import os
import numpy as np
import imutils
from argparse import ArgumentParser
#from pathvalidate.argparse import validate_filename_arg, validate_filepath_arg

import numpy as np
import cv2
 
def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
 

cwd = os.getcwd()
parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--out", type=str)
options = parser.parse_args()

if options.path:
    print("path: {}".format(options.path))
    folder = options.path
else:
    folder = cwd

if options.out:
    print("out: {}".format(options.out))
    outfolder = options.out
else:
    outfolder = cwd
#applyFilter(folder)


if __name__ == '__main__':
    files = os.listdir(folder)
    for i in range(0,len(files)):
        img_name, img_extension = os.path.splitext(files[i])
        if img_extension == '.jpg':
            #print(files[i])
            filename = files[i]

            img = cv2.imread(os.path.join(folder,files[i]))

            filters = build_filters()
            
            #print (filters)

            res1 = process(img, filters)
            #cv2.imshow('result', res1)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            height = res1.shape[0]
            width = res1.shape[1]
            #dim = (height, width)
            #print (dim)
            
            #resized = cv2.resize(res1, dim, interpolation = cv2.INTER_AREA)
            dst = cv2.Canny(img,100,200, apertureSize = 3)
            cv2.imwrite(outfolder + '/' + img_name + '_dst' + img_extension, dst)
            blank_image = np.zeros((height,width,3), np.uint8)
            blank_image = 0.4*blank_image + 0.6*res1


            # Copy edges to the images that will display the results in BGR
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

            #lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 500)
            linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, None, 50, 10)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (20, int(blank_image.shape[0]-20))
            # fontScale
            fontScale = 2
            # Blue color in BGR
            color = (0, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            if linesP is not None:
                blank_image = cv2.putText(blank_image, 'HL_count: ' + str(len(linesP)), org, font, fontScale, color, thickness, cv2.LINE_AA)


            # Draw the lines
            
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(blank_image, (l[0], l[1]), (l[2], l[3]), (255,255,0), 2, cv2.LINE_AA)

                cv2.imwrite(outfolder + '/' + img_name + '_edges' + img_extension, blank_image)

            filepath = str(outfolder + '/' + img_name + '_gabor' + img_extension)
            cv2.imwrite(filepath, res1)