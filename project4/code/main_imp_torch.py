import numpy as np
import cv2
from skimage.measure import compare_psnr,compare_ssim

np.random.seed(333)
def convert_rgb_to_gray(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    return 0.21*r+0.72*g+0.07*b
import matplotlib.pyplot as plt
class Convolution():
    def __init__(self,channel_in,channel_out,size=3,stride=2,padding=1):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.size = size
        self.stride = stride
        self.padding = padding
        self.weight = np.random.normal(0,1,(self.channel_in,self.channel_out,self.size,self.size))
        self.add = np.empty((self.channel_out,))

    def forward(self,x):
        B,C,W,H = x.shape
        self.w = W
        self.h = H
        self.b=B
        x_ = np.pad(x,((0,0),(0,0),(1,1),(1,1)),'constant')
        out =   np.empty( (B,self.channel_out,W//self.stride,H//self.stride) )
        for i in range(self.channel_out):
            conv = self.weight[:,i,:,:]
            for b in range(B):
                for j in range(1,W+1,self.stride):
                    for k in range(1,H+1,self.stride):
                        out[b,i,(j-1)//self.stride,(k-1)//self.stride] = np.sum(x_[b,:,j-1:j+2,k-1:k+2]*conv) #+ self.add[i]
        self.out = out
        return 1/(1+np.exp(-out))

    def backward(self,x,out,gt):
        self.grad_weight = np.zeros((self.channel_in,self.channel_out,self.size,self.size))
        x_ = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant')
        delta = 2*(out-gt)/(self.w*self.h)/2 * 1/(1+np.exp(-self.out)) *(1-1/(1+np.exp(-self.out)))
        # print('delta',delta.shape)
        # self.grad_add = np.ones((self.channel_out)) * delta

        for b in range(self.b):
            for i in range(self.channel_out):
                for j in range(1,self.w+1):
                    for k in range(1,self.h+1):
                        for c in range(self.channel_in):
                            for m in range(-1,2):
                                for n in range(-1,2):
                                    self.grad_weight[c,i,m+1,n+1] += delta[b,i,j-1,k-1] * (x_[b,c,j+m,k+n])
        print(self.grad_weight.max(),self.grad_weight.min())

    def update(self,lr):
        self.weight = self.weight - lr* self.grad_weight
        # self.add = self.add - lr * self.grad_add

class Sigmoid():
    def __init__(self):
        pass
    def forward(self,x):
        return 1/(1+np.exp(-x))
    def backward(self):
        pass


class Net():
    def __init__(self):
        pass

sz = 256
img_bgr = cv2.imread('wallhaven-vmm37p.jpg')
img_bgr = cv2.resize(img_bgr,(sz,sz))
# cv2.imwrite('img_%d.png'%sz,img_bgr)
img_rgb = img_bgr[:,:,::-1]
img_gray_o = convert_rgb_to_gray(img_rgb)
#
img_lab = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2LAB)
# plt.imshow(img_rgb[:,:,0],'gray')
# plt.show()
# plt.imshow(img_rgb[:,:,1],'gray')
# plt.show()
# plt.imshow(img_rgb[:,:,2],'gray')
# plt.show()
# print(img_rgb.shape)
# print(img_rgb[:,:,0].max(),img_rgb[:,:,1].max(),img_rgb[:,:,2].max())

#
w,h,_ = img_rgb.shape

# x = np.ones((1,1,256,256))
c2 = Convolution(1,3)
c1 = Convolution(1,2,stride=1)
img_gray = img_gray_o / 255
# img_gray = (img_gray - np.mean(img_gray)) / np.std(img_gray)
img_gray_ = np.expand_dims(np.expand_dims(img_gray,0),0)

img_lab = img_lab / 255
# img_rgb = (img_rgb - np.mean(img_rgb)) / np.std(img_rgb)
img_lab = img_lab.transpose(2,0,1)
img_lab_ = np.expand_dims(img_lab,0)

# yy = img_rgb * 255
# plt.imshow(yy[0].transpose(1, 2, 0).astype(np.uint8))
# plt.show()

if 1:
    import torch
    import torch.nn as nn


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.c1 = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 2, 3, stride=1, padding=1)

            )

        def forward(self, x):
            return self.c1(x)


    c1 = Net() # nn.Conv2d(1,3,3,padding=1) #
    opt = torch.optim.SGD(c1.parameters(), lr=0.01)
    # a = np.random.randint(0,250-64)
    # b = np.random.randint(0,250-64)
    # img_gray = torch.from_numpy(img_gray[:,:,a:a+64,b:b+64])
    # img_rgb = torch.from_numpy(img_rgb[:,:,a:a+64,b:b+64])
    sig = nn.Sigmoid()
    for i in range(10000):
        pz = 256
        a = 0#np.random.randint(0, 256 - pz)
        b = 0#np.random.randint(0, 256 - pz)
        img_gray = img_gray_ #[:, :, a:a + pz, b:b + pz]
        img_lab = img_lab_ #[:, :, a:a + pz, b:b + pz]
        # if np.random.rand() > 0.5:
        #     img_gray = img_gray[:, :, ::-1, :].copy()
        #     img_lab = img_lab[:, :, ::-1, :].copy()
        # if np.random.rand() > 0.5:
        #     img_gray = img_gray[:, :, :, ::-1].copy()
        #     img_lab = img_lab[:, :, :, ::-1].copy()

        img_gray = torch.from_numpy(img_gray)
        img_lab = torch.from_numpy(img_lab)

        y1 = c1(img_gray[:,:,:,0:128].float())
        y1 = sig(y1)
        # c1.backward(img_gray[:,:,:,0:128],y1,img_lab[:,1:3,:,0:128])
        # c1.update(0.1)


        loss = torch.mean((y1 - img_lab[:,1:3,:,0:128].float()) ** 2)
        loss.backward()
        opt.step()

        if i % 1000 == 0:
            print('epoch %d: %.4f' % (i, loss))
            ## pred
            with torch.no_grad():
                y1 = c1(img_gray[:,:,:,128:256].float())
                y1 = sig(y1)

                yy = y1.data.numpy() * 255
                yy2 = img_gray[:,:,:,128:256].data.numpy() * 255
                pred_lab= np.concatenate((yy2[0].transpose(1, 2, 0).astype(np.uint8),yy[0].transpose(1, 2, 0).astype(np.uint8)),2)
                pred_rgb = cv2.cvtColor(pred_lab,cv2.COLOR_LAB2RGB)

                pp = np.hstack((img_rgb[:,0:128,:], pred_rgb))
                plt.xticks([]), plt.yticks([])
                plt.imshow(pp)
                plt.savefig('256/CNN/%d.png'%i,dpi=300)

                #
                # ## input gray
                # yy = img_gray[:,:,:,128:256].data.numpy() * 255
                # plt.subplot(132)
                # plt.imshow(yy[0, 0].astype(np.uint8), 'gray')

                ## gt rgb
                yy = img_lab[:,:,:,128:256].data.numpy() * 255
                l = yy[0].transpose(1, 2, 0).astype(np.uint8)
                pp = cv2.cvtColor(l,cv2.COLOR_LAB2RGB)


                PSNR = compare_psnr(pp, pred_rgb,255)
                SSIM = compare_ssim(pp, pred_rgb, multichannel=True)
                print('psnr: ', PSNR)
                print('ssim: ', SSIM)




# for i in range(10):
#
#     y1 =c1.forward(img_gray)
#     yy= y1*255
#     plt.imshow(yy[0].transpose(1,2,0).astype(np.uint8))
#     plt.show()
#     loss = np.mean((y1-img_rgb)**2)
#     c1.backward(img_gray,y1,img_rgb)
#     c1.update(0.1)
#     print('epoch %d: %.4f'%(i,loss))

# y2 =c2.forward(x)
# print(y1.shape)
# print(y2.shape)
