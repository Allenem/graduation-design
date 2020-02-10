import os,sys
import magic
from PIL import Image, ImageChops, ImageEnhance

def brightenDiff(img1_path,img2_path,output_path):
  print("****brightenDiff is starting****")
  if magic.from_file(img1_path, mime=True) == "image/jpeg" and magic.from_file(img2_path, mime=True) == "image/jpeg" :
    quality_level = 85
    (img1realname, extension) = os.path.splitext(os.path.basename(img1_path))
    (img2realname, extension) = os.path.splitext(os.path.basename(img2_path))
    brighten_img_path = os.path.join(output_path,img1realname+"AND"+img2realname+"_diff.jpg")

    # get the differences between image & tmp_image, then brighten them
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)
    brighten_image = ImageChops.difference(image1, image2)
    extrema = brighten_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = max_diff
    brighten_image = ImageEnhance.Brightness(brighten_image).enhance(scale)

    brighten_image.save(brighten_img_path)
    print("****brightenDiff has been completed****")
  else:
    print("brightenDiff works only with JPEG")

if __name__ == "__main__":
  img1_path = "./img/books.jpg"
  img2_path = "./img/books-edited.jpg"
  output_path = "./img"
  brightenDiff(img1_path,img2_path,output_path)