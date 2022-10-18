import os, re, os.path
import imageio


def clearFiles():
    mypath = "data/"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            if file[0] != ".":
                os.remove(os.path.join(root, file))


def makeGIF():
    # Build GIF
    # with imageio.get_writer("data/animation.gif", mode="I") as writer:
    #     for root, dirs, files in os.walk("data"):
    #         for file in files:
    #             if file[0] != ".":
    #                 image = imageio.imread("data/{}".format(file))
    #                 writer.append_data(image)
    frames = []
    for root, dirs, files in os.walk("data"):
        files.sort()
        for file in files:
            if file.endswith(".png"):
                image = imageio.imread("data/{}".format(file))
                frames.append(image)
    imageio.mimsave("data/animation.gif", frames, format="GIF", duration=0.3)
