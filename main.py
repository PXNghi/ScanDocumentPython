import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract
import transform
import poly_editor as poly_f
from matplotlib.patches import Polygon
import img2pdf
from PIL import Image
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import filedialog

# Hay doc file README de biet them mot so luu y ve chuong trinh a

SAMPLE_DIR = 'samples/'
OUTPUT_DIR = 'outputs/'
PDF_DIR = 'pdfs/'
TXT_DIR = 'txts/'

def select_image():
    image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", ".jpg .jpeg .png .bmp")])
    global image_name
    image_name = os.path.basename(image_path)
    if image_path:
        check_file_existed(dir=PDF_DIR, image_path=image_name, ext='pdf')
        check_file_existed(dir=TXT_DIR, image_path=image_name, ext='txt')
        wrap_image = processContours(cv2.imread(image_path))
        global root
        root.destroy()
        itotext(wrap_image, image_name)
        # Gọi hàm xử lý ảnh tại đây
        print("Ảnh đã được chọn:", image_path)
    else:
        print("Ảnh không hợp lệ")


def resizeImage(image, width = 500):
    # get width and height of image
    h,w,c = image.shape
    # aspect ratio
    height = int((h / w) * width)
    size = (width, height)
    image = cv2.resize(image, (width, height))
    return image, size

def interactive_get_contour(screenCnt, image):
    poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    ax.set_title(('Kéo các góc để định vị tài liệu.'))
    p = poly_f.PolygonInteractor(ax, poly)
    plt.imshow(image)
    plt.show()

    new_points = p.get_poly_points()[:4]
    new_points = np.array([[p] for p in new_points], dtype = "int32")
    return new_points.reshape(4, 2)

def getContours(orig_image):
    # image, size = resizeImage(orig_image.copy())
    image = orig_image
    size = orig_image.shape # h w c
    detail_img = cv2.detailEnhance(image, sigma_s=20, sigma_r=0.15)
    gray_img = cv2.cvtColor(detail_img, cv2.COLOR_BGR2GRAY)
    _, threshold= cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(threshold, (5,5), 0)
    edge_img = cv2.Canny(blur, 0, 50)
    # morphological transform
    kernel = np.ones((5, 5), np.uint8)
    # increase the edges with thicker line is 1
    dilate = cv2.dilate(edge_img, kernel, iterations=1)
    # close the gap between each item
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    approx_contours = []
    # find the contours
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 80, True)

        if len(approx) == 4:
            approx_contours.append(approx)
            break
    
    if not approx_contours:
            TOP_RIGHT = (size[1], 0)
            BOTTOM_RIGHT = (size[1], size[0])
            BOTTOM_LEFT = (0, size[0])
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

    else:
        screenCnt = max(approx_contours, key=cv2.contourArea)
    
    return screenCnt.reshape(4,2)

def processContours(orig_image):
    imageCnt = getContours(orig_image)
    # img = resizeImage(orig_image)[0]
    img = orig_image
    screenCnt = interactive_get_contour(screenCnt=imageCnt, image=img)
    wrap_image = transform.four_point_transform(image=img, pts=screenCnt)
    # convert the warped image to grayscale
    gray = cv2.cvtColor(wrap_image, cv2.COLOR_BGR2GRAY)
    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    # apply adaptive threshold to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    # save the transformed image
    save_file_to_dir(image_name, thresh)

    return thresh

def save_file_to_dir(image_path, image):
    basename = os.path.basename(image_path)
    cv2.imwrite(OUTPUT_DIR + basename, image)
    cv2.imwrite(PDF_DIR + basename, image)

def replace_extension(text, ext):
    text_index = text.find('.')
    text = text[:text_index + 1] + ext
    return text

def check_file_existed(dir, image_path, ext):
    image_path = replace_extension(image_path, ext)
    if os.path.isfile(dir + image_path):
        os.remove(dir + image_path)

def img_to_pdf(image_path):
    print("path: " + image_path)
    image = Image.open(image_path)
    pdf_bytes = img2pdf.convert(image.filename)
    file = open(image_path, 'wb')
    file.write(pdf_bytes)
    image.close()
    file.close()
    p = replace_extension(image_path, 'pdf')
    os.rename(image_path, p)
    print("File PDF đã được tạo thành công. Mời kiểm tra thư mục pdfs.")
    return p

def write_to_txt(text, image_path):
    display_text_in_window(text=text, title='Original') 
    text = translate_text(text)
    display_text_in_window(text=text, title='Translated') 
    file = open(TXT_DIR + image_path, "x", encoding='utf-8')
    file.write(text)
    file.close()
    image_path = TXT_DIR + image_path
    p = replace_extension(image_path, 'txt')
    os.rename(image_path, p)

def translate_text(text):
    translated = GoogleTranslator(source='auto', target='vi').translate(text)
    return translated

def itotext(image, image_path):
    print("image_path: " + image_path)
    pytesseract.pytesseract.tesseract_cmd = 'c:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    text = pytesseract.image_to_string(image)
    write_to_txt(text=text, image_path=image_path)
    img_to_pdf(PDF_DIR + image_path)
    return text

def display_text_in_window(text, title):
    root = tk.Tk()
    root.title(title)
    text_widget = tk.Text(root)
    text_widget.insert("1.0", text)
    text_widget.pack()
    root.mainloop()

def main():
    global root
    root = tk.Tk()
    root.title("Chọn ảnh")

    button = tk.Button(root, text="Chọn ảnh", command=select_image)
    button.pack(padx=10, pady=10)

    root.mainloop()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__  == '__main__':
    main()