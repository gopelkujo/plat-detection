# import needed package
import cv2, tkinter, re, numpy as np, pytesseract
from PIL import ImageTk, Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from tkinter import Text, font as tkfont

# constant variable
canvas_size_x = 200
canvas_size_y = 200
window_size_x = 600
window_size_y = 450
custom_config = r'--oem 3 --psm 6'

# set up window
root = tkinter.Tk()
root.title('Plat Detection')
root.configure(background='#202020')
# root.attributes('-zoomed', True)
root.geometry(str(window_size_x) + 'x' + str(window_size_y))

def imageToCv2(img):
    img_cv = img
    img_cv = img_cv.convert('RGB')
    img_cv = np.array(img_cv)
    img_cv = img_cv[:, :, ::-1].copy()
    return img_cv

def cv2ToImage(cvimg):
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(cvimg)
    imgmerge = cv2.merge((r,g,b))
    imarray = Image.fromarray(imgmerge)
    imarray.thumbnail((canvas_size_x, canvas_size_y), Image.ANTIALIAS)
    # imgfile = ImageTk.PhotoImage(image=imarray)
    return imarray

def svg2png(filename):
    drawing = svg2rlg(filename)
    renderPM.drawToFile(drawing, 'temp.png', fmt='PNG')
    img = renderPM.drawToPIL(drawing)
    return img

def greyScaleFilter(opencv_img):
    return cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

def gaussianBlurFilter(opencv_img):
    return cv2.GaussianBlur(opencv_img, (3,3), 0)

def medianBlurFilter(opencv_img):
    return cv2.medianBlur(opencv_img,5)

def tresholdFilter(opencv_img):
    return cv2.threshold(opencv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilationFilter(opencv_img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(opencv_img, kernel, iterations = 1)

def erosionFilter(opencv_img):
    kernel = np.ones((1,1),np.uint8)
    return cv2.erode(opencv_img, kernel, iterations = 1)

def openingImage(opencv_img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(opencv_img, cv2.MORPH_OPEN, kernel)

def closingImage(opencv_img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(opencv_img, cv2.MORPH_CLOSE, kernel)

def cannyFilter(opencv_img):
    return cv2.Canny(opencv_img, 100, 200)

def sobelFilter(opencv_img):
    x = cv2.Sobel(opencv_img, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(opencv_img, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absx, 0.5, absy, 0.5,0)

def kMeansSegmentation(opencv_img):
    rgb_im = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    twoDimage = rgb_im.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((rgb_im.shape))

def inverseImg(opencv_img):
    img = opencv_img
    (rows,cols, _) = img.shape

    for i in range(rows):
        for j in range(cols):
            (r,g,b) = img[i,j]
            r = 255 - r
            if(r > 255): r = 255
            if(r < 0): r = 0
            g = 255 - g
            if(g > 255): g = 255
            if(g < 0): g = 0
            b = 255 - b
            if(b > 255): b = 255
            if(b < 0): b = 0
            img[i,j] = [r, g, b]
    return img

def deskew(opencv_img):
    coords = np.column_stack(np.where(opencv_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = opencv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(opencv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def incIntensity(opencv_img, value):
    img = opencv_img
    (rows,cols, _) = img.shape

    for i in range(rows):
        for j in range(cols):
            (r,g,b) = img[i,j]
            r = r + value
            if(r > 255): r = 255
            g = g + value
            if(g > 255): g = 255
            b = b + value
            if(b > 255): b = 255
            img[i,j] = [r, g, b]
    
    return img

def letterCorrection(number):
    switcher = {
        '1': 'I',
        '2': 'Z',
        '3': 'B',
        '4': 'A',
        '5': 'B',
        '6': 'G',
        '7': 'T',
        '0': 'O'
    }
    return switcher.get(str(number))

# add frame to window
detection_process_1 = tkinter.Frame(root, bg='#202020')
detection_process_1.pack(side = tkinter.LEFT, padx=10)
# detection_process_1.pack_propagate(0)
detection_process_2 = tkinter.Frame(root, bg='#202020')
detection_process_2.pack(side = tkinter.LEFT, padx=10)
option_frame = tkinter.Frame(root, bg='#202020', height=500, width=160)
option_frame.pack(side = tkinter.LEFT, padx=10)
option_frame.pack_propagate(0)

myFont = tkfont.Font(size=8)

# File Section
select_file_frame = tkinter.Frame(option_frame, bg='#202020', highlightbackground='white', highlightthickness=1, width=156, height=63, pady=5)
select_file_frame.pack(side=tkinter.TOP, pady=25)
select_file_frame.pack_propagate(0)
file_name_label = tkinter.Label(select_file_frame, text='No file selected.', bg='#202020', fg='white', font=myFont).pack()
btn_select_file = tkinter.Button(select_file_frame, text='Select File', height=1, font=myFont, bd=0, highlightthickness=0).pack(pady=2)

# Result Section
result_frame = tkinter.Frame(option_frame, bg='#202020', highlightbackground='white', highlightthickness=1, width=156, height=63, pady=5)
result_frame.pack(side=tkinter.TOP)
result_frame.pack_propagate(0)
result_label = tkinter.Label(result_frame, text='Result', bg='#202020', fg='white', font=myFont).pack()
result_text = tkinter.Text(result_frame, height=1, width='15', bd=0, highlightthickness=0, font=myFont).pack()
# btn_save_result = tkinter.Button(result_frame, text='Save All Result', height=1, width=13, font=myFont, bd=0, highlightthickness=0).pack(pady=2)

btn_start = tkinter.Button(option_frame, text='Start Process', font=myFont, bd=0, highlightthickness=0, bg='#0083DB', fg='white', width=15, height=3).pack(side=tkinter.BOTTOM, pady=10)

# reading image
opencv_img = cv2.imread('img/plat-nomor-2-noise.jpg')
opencv_img = incIntensity(opencv_img, 10)

step_0_label = tkinter.Label(detection_process_1, text = 'Original Image (0)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_0_img = ImageTk.PhotoImage(cv2ToImage(opencv_img))
step_0_show = tkinter.Label(detection_process_1, image=step_0_img).pack(side = tkinter.TOP)

# applying grey scale filter
step_1_cvimg = greyScaleFilter(opencv_img)
step_1_label = tkinter.Label(detection_process_1, text = 'Greyscale Image (1)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_1_img = cv2ToImage(step_1_cvimg)
step_1_img = ImageTk.PhotoImage(step_1_img)
step_1_show = tkinter.Label(detection_process_1, image=step_1_img).pack(side = tkinter.TOP)

# applying gausiian blur filter
step_2_cvimg = gaussianBlurFilter(step_1_cvimg)
step_2_label = tkinter.Label(detection_process_1, text = 'Gaussian Blur Image (2)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_2_img = cv2ToImage(step_2_cvimg)
step_2_img = ImageTk.PhotoImage(step_2_img)
step_2_show = tkinter.Label(detection_process_1, image=step_2_img).pack(side = tkinter.TOP)

# applying otsu binarization
step_3_cvimg = tresholdFilter(step_2_cvimg)
step_3_label = tkinter.Label(detection_process_2, text='Otsu Binarization (3)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_3_img = cv2ToImage(step_3_cvimg)
step_3_img = ImageTk.PhotoImage(step_3_img)
step_3_show = tkinter.Label(detection_process_2, image=step_3_img).pack(side = tkinter.TOP)

# applying morphology transformation (erosion)
step_4_cvimg = erosionFilter(step_3_cvimg)
step_4_label = tkinter.Label(detection_process_2, text='Erosion (4)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_4_img = cv2ToImage(step_4_cvimg)
step_4_img = ImageTk.PhotoImage(step_4_img)
step_4_show = tkinter.Label(detection_process_2, image=step_4_img).pack(side = tkinter.TOP)

step_5_cvimg = cv2.findContours(step_4_cvimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
step_5_cvimg = step_5_cvimg[0] if len(step_5_cvimg) == 2 else step_5_cvimg[1]
step_5_cvimg = sorted(step_5_cvimg, key = cv2.contourArea, reverse = True)

for c in step_5_cvimg:
    x, y, w, h = cv2.boundingRect(c)
    ROI = step_4_cvimg[y:y+h, x:x+w]
    break

step_5_label = tkinter.Label(detection_process_2, text='Cropped (6)', bg='#202020', fg='white').pack(side = tkinter.TOP)
step_5_img = cv2ToImage(ROI)
step_5_img = ImageTk.PhotoImage(step_5_img)
step_5_show = tkinter.Label(detection_process_2, image=step_5_img).pack(side = tkinter.TOP)

ocr_result = pytesseract.image_to_string(ROI, config=custom_config)

if(re.search("^[A-Z]", ocr_result) == False):
    to_list = list(ocr_result)
    to_list[0] = letterCorrection(to_list[0])
    ocr_result = "".join(to_list)
    print("Fixed format!")
else:
    print("Checked format!")

print(ocr_result.upper())

# make window stay
root.mainloop()