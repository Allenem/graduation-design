import cv2
import matplotlib.pyplot as plt


def draw_color_histogram(inputpath, outputpath):
    image = cv2.imread(inputpath)
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')

    plt.figure()                # new a figure
    plt.title("RGB Histogram")  # title of figure
    plt.xlabel("Bins")          # xlabel:lightness 0-255
    plt.ylabel("# of Pixels")   # ylabel:# of pixels of one lightness

    for (channels, color) in zip(channels, colors):
        hist = cv2.calcHist([channels], [0], None, [256], [0, 255])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.savefig(outputpath)     # save the figure as a picture
    # plt.show()                # show the figure
    plt.close()                 # close a figure window
