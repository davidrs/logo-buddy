import os
import os.path as op
import numpy as np
from glob import glob
import cv2
from .utils import OUT_DIR


def stickerfy(starter_logo_path, fun_logo_path):
    # apply the mask to the fun logo
    fun_logo = cv2.imread(fun_logo_path)

    # take starter logo and convert to binary mask
    starter_logo = cv2.imread(starter_logo_path, 0)

    ret, starter_logo = cv2.threshold(starter_logo, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite("starter_logo.png", starter_logo)
    # resize starter logo to match fun logo
    starter_logo = cv2.resize(starter_logo, (fun_logo.shape[1], fun_logo.shape[0]))
    mask = starter_logo > 0
    mask = mask.astype(np.uint8) * 255
    # save mask
    # cv2.imwrite("mask.png", mask)

    # invert the mask
    mask = cv2.bitwise_not(mask)

    # dilate the mask 5px
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.dilate(mask,kernel,iterations = 2)

    # use mask to brighten fun_log slightly, 0 values in mask should have no effect on fun lo
    # convertScaleAbs

    fun_logo_bright = cv2.convertScaleAbs(fun_logo, alpha=1.3, beta=0)
    fun_logo_bright = cv2.bitwise_and(fun_logo_bright, fun_logo_bright, mask=mask)
    # cv2.imwrite("fun_logo_bright.png", fun_logo_bright)
    # invert mask
    mask = cv2.bitwise_not(mask)
    fun_logo = cv2.bitwise_and(fun_logo, fun_logo, mask=mask)
    # cv2.imwrite("fun_logo.png", fun_logo)

    # combine
    fun_logo = cv2.add(fun_logo, fun_logo_bright)

    # return the fun logo
    return fun_logo


def stickerfy_all(starter_logo_path):
    """
    Takes already rendered images, and makes them better stickers with some contrast etc,
    """

    basename = op.basename(starter_logo_path).split(".")[0]
    basedir = op.join(OUT_DIR, basename)

    sticker_dir = op.join(basedir, "stickers")
    os.makedirs(sticker_dir, exist_ok=True)

    # get all the fun logos
    fun_logos = glob(op.join(basedir, "*png"))
    print("fun_logos", fun_logos)

    for fl in fun_logos:
        flname = op.basename(fl)
        sticker = stickerfy(starter_logo_path, fl)
        print("save to", op.join(sticker_dir, flname.replace(".png", "_sticker.png")))
        cv2.imwrite(
            op.join(sticker_dir, flname.replace(".png", "_sticker.png")), sticker
        )


if __name__ == "__main__":
    for input_img in (
        glob("input/skip/*jpeg")
        + glob("input/*png")
        + glob("input/skip/*png")
        + glob("input/*jpeg")
    ):
        sticker = stickerfy_all(input_img)
