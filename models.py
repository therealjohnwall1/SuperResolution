import torch
import torch.nn as nn
import torchvision.models as models
    
class VGG():
    def __init__(self):
        self.VGG = models.vgg19(pretrained = True)
        self.VGG.eval()
    
    def predict(self, input):
        return self.VGG(input)


#input low res -> #high res
class Generator(nn.Module):
    def __init__(self,res_layers, up_layers):
        self.res_layers = res_layers
        self.up_layers = up_layers
    
    def residual_block(self, x):
        res = nn.Conv2d(x.shape[1], 64, 3, 1, 1)(x)
        res = nn.BatchNorm2d(x.shape[1], momentum=0.5)(res)
        res = nn.Conv2d(x.shape[1], 64, 3, 1, 1)(x)
        res = nn.BatchNorm2d(x.shape[1], momentum=0.5)(res)
        return nn.add(x,res)
    
    def upscale_block(self, x):
        ups = nn.Conv2d(x.shape[1],256,3,1,1)(x) #conv + upsample = deconv
        ups = nn.Upsample(size = 2)(ups)
        ups = nn.PReLU()(ups)
        return ups
    
    def forward(self,x):
        x = nn.Conv2d(x.shape[1],64,9,1,1)(x)
        temp = x

        for i in range(self.res_layers):
            x = self.residual_block(x)

        x = nn.Conv2d(x.shape[1],64,3,1,1)(x)
        x = nn.BatchNorm2d(x.shape[1], momentum=0.5)(x)
        x = nn.add(x,temp)

        for i in range(self.up_layers):
            x = self.upscale_block(x)  
        return x
        

#discrimnator arch
    #disc block
        #conv,batch_norm,leaklu
class Discriminator(nn.Module):
    def __init__(self,disc_layers):
        self.disc_layers = disc_layers

    def discriminator_block(self,x,output,strides = 1, batch_norm=True): # batch_norm on = training
        disc = nn.Conv2d(x.shape[1],output,3, strides,1)(x)
        if batch_norm:
            disc = nn.BatchNorm2d(output, momentum = 0.8)(disc)
        disc = nn.LeakyReLU(0.2)(disc)
        return disc
    
    def forward(self,x):

        layer_op = 64

        l1 = self.discriminator_block(x,l1,batch_norm=False)
        l2 = self.discriminator_block(l1,layer_op,strides=2)
        l3 = self.discriminator_block(l2,layer_op*2)
        l4 = self.discriminator_block(l3,layer_op*2,strides=2)
        l5 = self.discriminator_block(l4,layer_op*4)
        l6 = self.discriminator_block(l5,layer_op*4,strides=2)
        l7 = self.discriminator_block(l6,layer_op*8)
        l8 = self.discriminator_block(l7,layer_op*4,strides=2)

        l8 = l8.view(-1)
        l9 = nn.linear(l8,layer_op*64)
        l10 = nn.LeakyReLU(0.2)(l9)
        l11 = nn.linear(l10,1)
        validity = nn.Sigmoid(l11)

        return validity

        





        



