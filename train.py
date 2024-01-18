from models import *
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import torchvision.models as tmodels
import numpy as np
from PIL import Image
import random
from torch.autograd import Variable

set = 500

print("processing low rates")
lrs = os.listdir("data/low_res")[:set]

lr_imgs = []
for images in tqdm(lrs):
    img_lr = Image.open("data/low_res/"+ images)
    lr_arr = np.array(img_lr)
    lr_imgs.append(lr_arr)

lr_arr = np.asarray(lr_imgs)


print("processing high rates")
hrs = os.listdir("data/high_res")[:set]

hr_imgs = []
for images in tqdm(lrs):
    img_hr = Image.open("data/high_res/"+ images)
    hr_arr = np.array(img_hr)
    hr_imgs.append(hr_arr)

hr_arr = np.asarray(hr_imgs)

#normalization
lr_arr = lr_arr/255.
hr_arr = hr_arr/255.

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_arr, hr_arr, 
                                                      test_size=0.15, random_state=69)

cuda = torch.cuda.is_available()
batch_size = 1
EPOCHS = 8
#create models

vgg19_model = tmodels.vgg19(pretrained=True)

generator = Generator(16,2) #16 residual layers, 2 upscales
discriminator = Discriminator(64) 
feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]) #shallow version of vgg
feature_extractor.eval() #no grads

criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
#assume cuda
generator = generator.cuda()
discriminator = discriminator.cuda()
feature_extractor = feature_extractor.cuda()
criterion_GAN = criterion_GAN.cuda()
criterion_content = criterion_content.cuda()

lr = 8e-5
b1 = 0.5
b2 = 0.999

optim_g = torch.optim.Adam(generator.parameters(), lr = lr,betas =(b1,b2))
optim_d = torch.optim.Adam(discriminator.parameters(), lr = lr,betas =(b1,b2))



Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#how does training loop work
#train discrim -> give it real and fake(generator) images
#find loss ()
#disable discrim train
#train generator using vgg
#we use our combined model



train_gen_losses, train_disc_losses, train_counter = [], [], []
test_gen_losses, test_disc_losses = [], []
for epoch in range(EPOCHS):
    # Training
    generator.train()
    discriminator.train()
    total_gen_loss = 0
    total_disc_loss = 0

    for i in range(len(lr_train)):
        imgs_lr = Variable(Tensor(lr_train[i]).unsqueeze(0))
        imgs_hr = Variable(Tensor(hr_train[i]).unsqueeze(0))

        optim_g.zero_grad()

        # Generate a high-resolution image
        gen_hr = generator(imgs_lr)
        discriminator.bn = True

        # Adversarial loss
        valid = Variable(Tensor(np.ones((1, *discriminator.output_shape))), requires_grad=False)
        adversarial_loss = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        content_loss = criterion_content(gen_features, real_features.detach())

        gen_loss = adversarial_loss + 0.006 * content_loss

        discriminator.bn = False

        # Backpropagation
        gen_loss.backward()
        optim_g.step()

        optim_d.zero_grad()

        #get validity
        real_loss = criterion_GAN(discriminator(imgs_hr), valid)
        fake_loss = criterion_GAN(discriminator(gen_hr.detach()), Variable(Tensor(np.zeros((1, *discriminator.output_shape))), requires_grad=False))
        disc_loss = 0.5 * (real_loss + fake_loss)

        # Backpropagation
        disc_loss.backward()
        optim_d.step()

        # Accumulate losses
        total_gen_loss += gen_loss.item()
        total_disc_loss += disc_loss.item()

    # Calculate average losses
    avg_gen_loss = total_gen_loss / len(lr_train)
    avg_disc_loss = total_disc_loss / len(lr_train)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

    # Save losses for plotting
    train_gen_losses.append(avg_gen_loss)
    train_disc_losses.append(avg_disc_loss)

