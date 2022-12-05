import os
import os.path
import imageio


def clearFiles():
    mypath = "data/"
    for root, _, files in os.walk(mypath):
        for file in files:
            if file[0] != ".":
                os.remove(os.path.join(root, file))


def makeGIF():
    frames = []
    for _, _, files in os.walk("data"):
        files.sort()
        for file in files:
            if file.endswith(".png"):
                image = imageio.imread("data/{}".format(file))
                frames.append(image)
    imageio.mimsave("data/animation.gif", frames, format="GIF", duration=0.3)
