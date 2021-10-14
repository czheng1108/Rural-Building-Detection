import cv2
import os


def image_cropping(image_path, image_to_path, image_size, image_to_size):
    img = cv2.imread(image_path)
    cropped = []
    for i in range(image_size//image_to_size):
        for j in range(image_size//image_to_size):
            cropped.append(img[i*image_to_size:image_to_size*(i+1), j*image_to_size:image_to_size*(j+1)])
    i = 1
    for img_cropped in cropped:
        if img_cropped.size == 0:
            continue
        cv2.imwrite(image_to_path + '/' + image_path.split('/')[-1].split('.')[0] + '-' + str(i) + '.png', img_cropped)
        i = i + 1


if __name__ == '__main__':
    # image_path = 'data/original/dbc-1x-22.png'
    image_path = os.listdir('data/original')
    image_to_path = 'data/cropped'
    if not os.path.exists(image_to_path):
        os.mkdir(image_to_path)
    for img_path in image_path:
        img_path = os.path.join('data/original', img_path)
        image_cropping(img_path, image_to_path, 1000, 64)
