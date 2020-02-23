import os
import time
import datetime
from PIL import Image

log_path = 'G:/PGGAN_face/log.txt'
resize_x = 256
resize_y = 256

def transimg(inputfolder,outputfolder):
    names = os.listdir(inputfolder)
    nameslen = len(names)
    print('# of file in {} is : {}'.format(inputfolder,nameslen))
    pngcount = 0
    notpng = 0
    damaged = 0
    for name in names:
        inputpath = inputfolder + '/' + name 
        if os.path.getsize(inputpath)  != 0: 
            img = Image.open(inputpath)
            resized_img = img.resize((resize_x, resize_y))
            namelist = name.split(".")
            if namelist[-1] == "png":
                namelist[-1] = "jpg"
                namestr = str.join(".", namelist) 
                outputpath = outputfolder + '/' + namestr
                # r,g,b,a = resized_img.split()              
                # resized_img = Image.merge("RGB",(r,g,b)) 
                resized_img.save(outputpath)
                pngcount += 1
            else:
                notpng += 1
                continue
        else:
            damaged += 1
    with open(log_path, "a", encoding="utf-8") as fi:
        fi.write('\n'+inputfolder+' fileslen: '+str(nameslen)+' pngcount: '+str(pngcount)+' notpng: '+str(notpng)+' damaged: '+str(damaged))


if __name__ == '__main__':

    start_time = time.clock()

    inputcommonpath = 'G:/PGGAN'
    outputcommonpath = 'G:/PGGAN_face'
    folders = [ 'devel' , 'test' , 'train' , 'img_pggan' ]

    for folder in folders:
        if folder != 'img_pggan' :
            inputfolder = inputcommonpath + '/' + folder 
            outputfolder = outputcommonpath + '/' + folder 
            if not os.path.exists(outputfolder):
                os.mkdir(outputfolder)
            # transimg(inputfolder,outputfolder)
        else:
            tempinfolder = inputcommonpath + '/' + folder 
            tempoutfolder = outputcommonpath + '/' + folder 
            if not os.path.exists(tempoutfolder):
                os.mkdir(tempoutfolder)
            subfolders = os.listdir(tempinfolder)
            for subfolder in subfolders:
                inputfolder = inputcommonpath + '/' + folder + '/' + subfolder 
                outputfolder = outputcommonpath + '/' + folder + '/' + subfolder
                if not os.path.exists(outputfolder):
                    os.mkdir(outputfolder)
                transimg(inputfolder,outputfolder)


    end_time = time.clock()
    delta_time = datetime.timedelta(seconds = (end_time-start_time))
    print('Running time is: %s '%(delta_time))