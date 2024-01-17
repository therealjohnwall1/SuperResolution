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
# test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, EPOCHS+1)]

for epoch in range(EPOCHS):

    ### Training
    gen_loss, disc_loss = 0, 0
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
    for batch_idx, imgs in enumerate(tqdm_bar):
        generator.train(); discriminator.train()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        ### Train Generator
        optim_g.zero_grad()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optim_g.step()

        ### Train Discriminator
        optim_d.zero_grad()
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optim_d.step()

        gen_loss += loss_G.item()
        train_gen_losses.append(loss_G.item())
        disc_loss += loss_D.item()
        train_disc_losses.append(loss_D.item())
        train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset))
        tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))

    # Testing
    gen_loss, disc_loss = 0, 0
    tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
    for batch_idx, imgs in enumerate(tqdm_bar):
        generator.eval(); discriminator.eval()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        ### Eval Generator
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        ### Eval Discriminator
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))
        
        # Save image grid with upsampled inputs and SRGAN outputs
        if random.uniform(0,1)<0.1:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
            save_image(img_grid, f"images/{batch_idx}.png", normalize=False)

    test_gen_losses.append(gen_loss/len(test_dataloader))
    test_disc_losses.append(disc_loss/len(test_dataloader))
    
    # Save model checkpoints
    if np.argmin(test_gen_losses) == len(test_gen_losses)-1:
        torch.save(generator.state_dict(), "saved_models/generator.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
        






