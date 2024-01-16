from PIL import Image
import numpy as numpy
import os
#downsize and upsize images -> split

path = "data/mirflickr"
for pic in (os.listdir(path)):
    img_raw = Image.open(path + "/" + pic)
    hr_img = img_raw.resize((128,128))
    lr_img = img_raw.resize((32,32))

    hr_img.save(path + "/high_res_imgs")
    lr_img.save(path + "/low_res_imgs")

    

