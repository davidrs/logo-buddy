import cv2

#  Sampler: DPM++ SDE Karras, CFG scale: 6,
def read_fit(img_path, max_width=768):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image to X width, keep ratio
    h, w, _ = image.shape
    new_w = max_width
    new_h = int(h * (new_w / w))
    image = cv2.resize(image, (new_w, new_h))
    return image
