import random
from PIL import Image

def white():
    return 255, 255, 255
def white_f():
    img = Image.new('RGB', (8, 8), white())
    # img.show()
    return img

def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
def random_color_f():
    img = Image.new('RGB', (8, 8), random_color())
    # img.show()
    return img


