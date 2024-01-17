from PIL import Image
import os
from tqdm import tqdm

path = "data/mirflickr"
os.makedirs(os.path.join("data/high_res"), exist_ok=True)
os.makedirs(os.path.join("data/low_res"), exist_ok=True)

for pic in tqdm(os.listdir(path)):
    img_raw = Image.open(os.path.join(path, pic))
    hr_img = img_raw.resize((128, 128))
    lr_img = img_raw.resize((32, 32))

    hr_img.save("data/high_res/" + pic, format="JPEG")
    lr_img.save("data/low_res/" + pic, format="JPEG")

