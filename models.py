""" 
File to contain differnt deep neural network models
"""
from torch import nn

# discriminator defined as a binary CNN based classifier, taken and modified from the following link
# https://learnopencv.com/paired-image-to-image-translation-pix2pix/
# Keep doing a batch wise
# this have 5 conv layers
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf = 64, n_layers = 3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 3 #make it odd
        padw = 1 #kw//2 to keep same
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                      stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        #remove the sigmoid and put it in the end
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw,
                               bias=False), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # print(self.model(input).shape)
        return self.model(input) #.mean() #taking mean as we only have to do the binary classification
        #I have removed the mean after discussing with marius as it is better to patch wise classification
