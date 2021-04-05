import cv2
import random
import numpy as np

# Augmentation
# Image
# - Noise : N(mean=1, std=0.05) * image
# - Gray : alpha(u(0.0, 1.0) * image + (1 - alpha) * gray
# - Brightness : (image + u(0, 32)) * u(0.5, 1.5)
# - Contrast : (image - mean(iamge)) * u(0.5, 1.5) + mean(image)
# - Color : (image - gray) * u(0.0, 1.0) + image
# - Equalization : convert to YCbCr -> Equalize Histogram of Y Channel
# - Sharpness : (image - f(image)) * u(0.0, 1.0) + image f : filter(1, 1, 1, 1, 5, 1, 1, 1, 1, ) / 13
# - Power-Law(Gamma) Transformation : gamma: u(0.8, 1.2) image = (image / 255)^gamma * 255
# - JPEG Compression Artifact : quality u(0, 100) save->load

# Random Rotation -15~+15
# Random Translation -5%~+5%
# Random Flipping 50%
# Random Rescaling -15%~+15%
# Random Gaussian Blur 50%, 5x5 Kernel sigma = 1, occlusion (50%, 16x16 Cutout)


def Rotation(input_tensor, label_tensor, degree):
    img = input_tensor[:,:,:3]
    h, w = img.shape[:2]

    zid = input_tensor[:,:,3:]

    # x = 255.0 * (x - x.min()) / (x.max() - x.min())
    # y = 255.0 * (y - y.min()) / (y.max() - y.min())
    # z = 255.0 * (z - z.min()) / (z.max() - z.min())
    #
    # intensity = 255.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
    # depth = 255.0 * (depth - depth.min()) / (depth.max() - depth.min())

    # cv2.imshow("img", img.astype(np.uint8))
    # # cv2.imshow("xyz", xyz.astype(np.uint8))
    # cv2.imshow("x", x.astype(np.uint8))
    # cv2.imshow("y", y.astype(np.uint8))
    # cv2.imshow("z", z.astype(np.uint8))
    # cv2.imshow("intensity", intensity.astype(np.uint8))
    # cv2.imshow("depth", depth.astype(np.uint8))

    new_degree = random.uniform(-degree, degree)

    M = cv2.getRotationMatrix2D((w/2, h/2), new_degree, 1)

    img = cv2.warpAffine(img, M, (w, h))
    zid = cv2.warpAffine(zid, M, (w, h))
    input_tensor = np.concatenate([img, zid], axis=-1)
    label_tensor = cv2.warpAffine(label_tensor, M, (w, h))
    label_tensor = np.expand_dims(label_tensor, axis=-1)

    return input_tensor, label_tensor

def Scale(input_tensor, label_tensor):
    img = input_tensor[:,:,:3]
    h, w = img.shape[:2]

    zid = input_tensor[:,:,3:]
    factor = random.uniform(0.6, 1.3)

    if (int)(w * factor) % 2 != 0:
        re_w = (int)(w * factor) + 1
    else:
        re_w = (int)(w * factor)
    if (int)(h * factor) % 2 != 0:
        re_h = (int)(h * factor) + 1
    else:
        re_h = (int)(h * factor)
    img = cv2.resize(img, (re_w, re_h), interpolation=cv2.INTER_LINEAR)
    zid = cv2.resize(zid, (re_w, re_h), interpolation=cv2.INTER_LINEAR)
    label_tensor = cv2.resize(label_tensor, (re_w, re_h), interpolation=cv2.INTER_LINEAR)

    scaled_h, scaled_w = img.shape[:2]
    if factor <= 1:
        udp = (int)((h - scaled_h ) / 2)
        lrp = (int)((w - scaled_w) / 2)
        img = cv2.copyMakeBorder(img, udp, udp, lrp, lrp, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        zid = cv2.copyMakeBorder(zid, udp, udp, lrp, lrp, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        label_tensor = cv2.copyMakeBorder(label_tensor, udp, udp, lrp, lrp, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        label_tensor = np.expand_dims(label_tensor, axis=-1)
    else:
        img = img[(int)(scaled_h/2 - h/2):(int)(scaled_h/2 + h/2), (int)(scaled_w/2 - w/2):(int)(scaled_w/2 + w/2),:]
        zid = zid[(int)(scaled_h/2 - h/2):(int)(scaled_h/2 + h/2), (int)(scaled_w/2 - w/2):(int)(scaled_w/2 + w/2),:]
        # label_tensor = label_tensor[(int)(scaled_h/2 - h/2):(int)(scaled_h/2 + h/2), (int)(scaled_w/2 - w/2):(int)(scaled_w/2 + w/2),:]
        label_tensor = label_tensor[(int)(scaled_h / 2 - h / 2):(int)(scaled_h / 2 + h / 2),
                       (int)(scaled_w / 2 - w / 2):(int)(scaled_w / 2 + w / 2)]
        label_tensor = np.expand_dims(label_tensor, axis=-1)

    input_tensor = np.concatenate([img, zid], axis=-1)

    return input_tensor, label_tensor

def Translate(input_tensor, label_tensor, tx, ty):
    img = input_tensor[:, :, :3]
    h, w = img.shape[:2]

    zid = input_tensor[:, :, 3:]

    new_tx = random.uniform(-tx, tx)
    new_ty = random.uniform(-ty, ty)

    M = np.float32([[1, 0, new_tx], [0, 1, new_ty]])

    img = cv2.warpAffine(img, M, (w, h))
    zid = cv2.warpAffine(zid, M, (w, h))

    input_tensor = np.concatenate([img, zid], axis=-1)
    label_tensor = cv2.warpAffine(label_tensor, M, (w, h))
    label_tensor = np.expand_dims(label_tensor, axis=-1)

    return input_tensor, label_tensor

def Flip(input_tensor, label_tensor, p):
    img = input_tensor[:, :, :3]
    h, w = img.shape[:2]

    zid = input_tensor[:, :, 3:]

    if p < random.random():
        img = cv2.flip(img, 1)
        zid = cv2.flip(zid, 1)
        input_tensor = np.concatenate([img, zid], axis=-1)
        label_tensor = cv2.flip(label_tensor, 1)
        label_tensor = np.expand_dims(label_tensor, axis=-1)
        return input_tensor, label_tensor
    else:
        return input_tensor, label_tensor

def White_Noise(img):
    h, w, c = img.shape
    mean = 1
    sigma = 0.1
    gauss = np.random.normal(mean, sigma, (h, w, c))
    gauss = gauss.reshape(h, w, c)

    noisy = img * gauss
    noisy = np.clip(noisy, 0, 255.0)
    noisy = noisy.astype('uint8')

    return noisy

def Gray(img):
    alpha = random.random()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = alpha * img + (1 - alpha) * gray
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def add_light(image):
    offset = random.uniform(0.0, 1.0)
    if offset>=0.5:
        gamma = random.uniform(1.5, 2.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        image = cv2.LUT(image, table)
        return image
    else:
        gamma = random.uniform(0.3, 0.7)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        image = cv2.LUT(image, table)
        return image

def contrast_image(image):
    contrast = random.randint(1, 25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def saturation_image(image):
    saturation = random.randint(50, 100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def Brightness(img):
    img = img.astype('float32')
    random_brightness = random.randint(0, 32)
    random_saturation = random.uniform(0.5, 1.5)

    img = (img + random_brightness) * random_saturation
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Contrast(img):
    mean = np.mean(img)
    random_contrast = random.uniform(0.5, 1.5)

    img = (img - mean) * random_contrast + mean
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Color(img):
    img = img.astype('float32')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = (img - gray) * random.random() + img
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_cbcr = cv2.split(img)
    hist, bins = np.histogram(y_cbcr[0], 256, [0, 256])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)

    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    y_cbcr[0] = cdf[y_cbcr[0]]
    img = cv2.merge(y_cbcr)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img

def Shapness(img):
    kernel_shapen = np.array([[1, 1, 1], [1, 9, 1], [1, 1, 1]]) / 11.0
    k_img = cv2.filter2D(img, -1, kernel_shapen)
    img = (img - k_img) * random.random() + img
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img

def Power_Law(img):
    gamma = random.uniform(0.8, 1.3)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img
    return img