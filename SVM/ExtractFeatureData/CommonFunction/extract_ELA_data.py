import os
import magic
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance


def extract_ELA_data(inputpath, outputpath, outputfile):
    if magic.from_file(inputpath, mime=True) == "image/jpeg":
        quality_level = 85
        (filerealname, extension) = os.path.splitext(os.path.basename(inputpath))
        tmp_path = os.path.join(outputpath, filerealname+"_tmp.jpg")
        ela_path = os.path.join(outputpath, filerealname+"_ela.jpg")

        image = Image.open(inputpath)
        # convert gray image
        image_gray = image.convert('L')
        # resave image
        image_gray.save(tmp_path, 'JPEG', quality=quality_level)

        # get the differences between image_gray & tmp_image, then save them pixel data
        tmp_image = Image.open(tmp_path)
        ela_image = ImageChops.difference(image_gray, tmp_image)
        extrema = ela_image.getextrema()
        max_diff = extrema[1]
        scale = 10*255/max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        width, height = ela_image.size[0], ela_image.size[1]
        pixel = []
        pixel_data = []
        for h in range(0, height):
            for w in range(0, width):
                pixel.append(ela_image.getpixel((w, h)))
            pixel_data.append(pixel)
            pixel = []
        print(len(pixel_data))

        dt = pd.DataFrame(pixel_data)
        sheet_name = inputpath.split('/')[-1][:-4]
        dt.to_excel(outputfile, sheet_name=sheet_name, index=0)

        # save ela , remove tmp
        # ela_image.save(ela_path)
        os.remove(tmp_path)
        # print("****ELA has been completed****")
    else:
        print("ELA works only with JPEG")
