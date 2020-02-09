import os,sys
import magic
from PIL import Image, ImageChops, ImageEnhance


def ela(filename, output_path):
  print("****ELA is starting****")
  if magic.from_file(filename, mime=True) == "image/jpeg":
    quality_level = 85
    # 获取文件名，.后缀名
    (filerealname, extension) = os.path.splitext(os.path.basename(filename))
    # print(filerealname, extension)
    tmp_img = os.path.join(output_path,filerealname+"_tmp.jpg")
    ela = os.path.join(output_path,filerealname+"_ela.jpg")
    image = Image.open(filename)
    image.save(tmp_img, 'JPEG', quality=quality_level)
    tmp_img_file = Image.open(tmp_img)
    ela_image = ImageChops.difference(image, tmp_img_file)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(ela)
    os.remove(tmp_img)
    print("****ELA has been completed****")
  else:
    print("ELA works only with JPEG")

if __name__ == "__main__":
  filename = "./img/butterfly.jpg"
  output_path = "./img"
  ela(filename, output_path)