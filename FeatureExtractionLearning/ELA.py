import os
import magic
# import BrightenDiff
from PIL import Image, ImageChops, ImageEnhance


def ela(filename, output_path):
  print("****ELA is starting****")
  if magic.from_file(filename, mime=True) == "image/jpeg":
    quality_level = 85
    (filerealname, extension) = os.path.splitext(os.path.basename(filename))
    tmp_path = os.path.join(output_path,filerealname+"_tmp.jpg")
    ela_path = os.path.join(output_path,filerealname+"_ela.jpg")

    # resave image
    image = Image.open(filename)
    image.save(tmp_path, 'JPEG', quality=quality_level)

    # get the differences between image & tmp_image, then brighten them
    tmp_image = Image.open(tmp_path)
    ela_image = ImageChops.difference(image, tmp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 5*255/max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # save ela , remove tmp
    ela_image.save(ela_path)
    os.remove(tmp_path)
    print("****ELA has been completed****")
  else:
    print("ELA works only with JPEG")
    
'''
# use BrightenDiff.brightenDiff
def ela(filename, output_path):
  print("****ELA is starting****")
  quality_level = 85
  (filerealname, extension) = os.path.splitext(os.path.basename(filename))
  tmp_path = os.path.join(output_path,filerealname+"_tmp.jpg")
  image = Image.open(filename)
  image.save(tmp_path, 'JPEG', quality=quality_level)
  BrightenDiff.brightenDiff(filename,tmp_path,output_path)
  os.remove(tmp_path)
  print("****ELA has been completed****")
'''

if __name__ == "__main__":
  filename = "./img/butterfly.jpg"
  output_path = "./img"
  ela(filename, output_path)