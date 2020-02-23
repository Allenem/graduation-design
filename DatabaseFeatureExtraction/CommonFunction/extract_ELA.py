import os
import magic
from PIL import Image, ImageChops, ImageEnhance

def draw_ELA(inputpath, outputpath):
    # print("****ELA is starting****")
    if magic.from_file(inputpath, mime=True) == "image/jpeg":
        quality_level = 85
        (filerealname, extension) = os.path.splitext(os.path.basename(inputpath))
        tmp_path = os.path.join(outputpath,filerealname+"_tmp.jpg")
        ela_path = os.path.join(outputpath,filerealname+"_ela.jpg")

        # resave image
        image = Image.open(inputpath)
        image.save(tmp_path, 'JPEG', quality=quality_level)

        # get the differences between image & tmp_image, then brighten them
        tmp_image = Image.open(tmp_path)
        ela_image = ImageChops.difference(image, tmp_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 10*255/max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # save ela , remove tmp
        ela_image.save(ela_path)
        os.remove(tmp_path)
        # print("****ELA has been completed****")
    else:
        print("ELA works only with JPEG")