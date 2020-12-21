import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
np.random.seed(3)
def convert_rgb_to_gray(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    return 0.21*r+0.72*g+0.07*b

class Sigmoid():
    def __init__(self):
        pass
    def __call__(self, x):
        return self.forward(x)
    def forward(self,x):
        self.x = x
        return 1/(1+np.exp(-x))
    def backward(self,out_grad):
        self.grad_x = 1/(1+np.exp(-self.x)) *(1-1/(1+np.exp(-self.x)))
        # print(self.grad_x)
        return self.grad_x * out_grad

class Convolution():
    def __init__(self,channel_in,channel_out,size=3,stride=2,padding=1):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.size = size
        self.stride = stride
        self.padding = padding
        self.weight = np.random.normal(0,1,(self.channel_in,self.channel_out,self.size,self.size))
        # self.add = np.empty((self.channel_out,))
    def __call__(self, x):
        res = self.forward(x)
        return res


    def forward(self,x):
        B,C,W,H = x.shape
        self.w = W
        self.h = H
        self.b=B
        self.c = C
        x_ = np.pad(x,((0,0),(0,0),(self.size//2,self.size//2),(self.size//2,self.size//2)),'constant')
        # print('x_: ',x_.shape)
        self.x_=x_
        out =   np.empty( (B,self.channel_out,W//self.stride,H//self.stride) )
        for i in range(self.channel_out):
            conv = self.weight[:,i,:,:]
            for b in range(B):
                for j in range(self.size//2,W+1,self.stride):
                    for k in range(self.size//2,H+1,self.stride):
                        out[b,i,(j-1)//self.stride,(k-1)//self.stride] = np.sum(x_[b,:,j-self.size//2:j+self.size//2+1,k-self.size//2:k+self.size//2+1]*conv) #+ self.add[i]
        self.out = out
        return out

    def backward(self,out_grad):
        self.grad_weight = np.zeros((self.channel_in,self.channel_out,self.size,self.size))
        self.grad_x = np.zeros(self.x_.shape) #(self.b,self.c,self.w,self.h)) #self.x_.shape
        # print(out_grad.shape)
        # print(self.grad_x.shape)
        x_ = self.x_
        delta = out_grad
        # delta = 2*(out-gt)/(self.w*self.h)/2 * 1/(1+np.exp(-self.out)) *(1-1/(1+np.exp(-self.out)))
        # print('delta',delta.shape)
        # self.grad_add = np.ones((self.channel_out)) * delta

        for b in range(self.b):
            for i in range(self.channel_out):
                for j in range(self.size//2,self.w+1,self.stride):
                    for k in range(self.size//2,self.h+1,self.stride):
                        for c in range(self.channel_in):
                            for m in range(int((1-self.size)/2),int((1+self.size)/2)):
                                for n in range(int((1-self.size)/2),int((1+self.size)/2)):
                                    self.grad_weight[c,i,m+1,n+1] += delta[b,i,(j-self.size//2)//self.stride,(k-self.size//2)//self.stride] * (x_[b,c,j+m,k+n])
                                    self.grad_x[b,c,j+m,k+n] += delta[b,i,(j-self.size//2)//self.stride,(k-self.size//2)//self.stride] * self.weight[c,i,m+1,n+1]
        # print(self.grad_weight.max(),self.grad_weight.min())
        return self.grad_x[:,:,1:self.w+1,1:self.h+1],self.grad_weight

    def update(self,lr):
        # print('weight: ',self.weight.max(),self.weight.min())
        # print('grad: ', self.grad_weight.max(), self.grad_weight.min())
        self.weight = self.weight - lr* self.grad_weight
        # self.add = self.add - lr * self.grad_add

class Upsampling():
    def __init__(self,scale):
        self.scale = scale
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        # print(x.repeat(self.scale,axis=2).repeat(self.scale,axis=3).shape)
        return x.repeat(self.scale,axis=2).repeat(self.scale,axis=3)
    def backward(self,out_grad):
        # print('grad_shape: ',out_grad[:,:,::self.scale,::self.scale].shape)
        return out_grad[:,:,::self.scale,::self.scale]

class Relu():
    def __init__(self):
        pass
    def __call__(self,x):
        return self.forward(x)
    def forward(self,x):
        self.x = x
        x_new = x*(x>0)
        return x_new
    def backward(self,out_grad):
        return out_grad*(self.x>0)

class Loss():
    def __init__(self):
        pass
    def __call__(self,pred,gt):
        res = self.forward(pred,gt)
        return res
    def forward(self,pred,gt):
        self.pred = pred
        self.gt = gt
        return np.mean((pred-gt)**2)
    def backward(self):
        b,c,w,h = self.pred.shape
        self.grad_x = 2.*(self.pred-self.gt)/c/w/h
        return self.grad_x

class Net():
    def __init__(self):
        cc=2
        self.conv = Convolution(1,cc,stride=2)
        self.relu1 = Relu()
        self.conv2 = Convolution(cc, cc, stride=1)
        self.relu2 = Relu()
        self.up = Upsampling(2)
        self.conv3 = Convolution(cc,cc,stride=1)
        self.relu3 = Relu()
        self.conv4 = Convolution(cc, 2, stride=1)
        self.sig = Sigmoid()
        self.loss =Loss()
    def __call__(self, x,gt):
        out,loss = self.forward(x,gt)
        return out,loss
    def forward(self,x,gt):
        x1 = self.relu1(self.conv(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.up(x2)
        x4 = self.relu3(self.conv3(x3))
        x5 = self.sig(self.conv4(x4))
        # print(x1.shape,x2.shape)
        loss = self.loss(x5,gt)
        return x5,loss
    def backward(self):
        grad_pred = self.loss.backward()
        grad_sig = self.sig.backward(grad_pred)
        grad_conv4,grad_weight4 = self.conv4.backward(grad_sig)
        grad_relu3 = self.relu3.backward(grad_conv4)
        grad_conv3, grad_weight3 = self.conv3.backward(grad_relu3)
        grad_up = self.up.backward(grad_conv3)
        grad_relu2 = self.relu2.backward(grad_up)
        grad_conv2, grad_weight2 = self.conv2.backward(grad_relu2)
        grad_relu = self.relu1.backward(grad_conv2)
        grad_conv, grad_weight = self.conv.backward(grad_relu)
        # print('grad_pred: ',grad_pred.shape)
        # print('grad_sig: ',grad_sig.shape)
        # print('grad_weight: ',grad_weight.shape)
    def update(self,lr):
        self.conv.update(lr)
        self.conv2.update(lr)
        self.conv3.update(lr)
        self.conv4.update(lr)

x1 = np.empty((1,1,256,256))
gt = np.empty((1,2,256,256))

sz = 256
img_bgr = cv2.imread('wallhaven-vmm37p.jpg')
img_bgr = cv2.resize(img_bgr,(sz,sz))
img_rgb = img_bgr[:,:,::-1]
img_gray = convert_rgb_to_gray(img_rgb)
img_lab = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2LAB)

train_img = img_gray[:,0:128]
train_gt = img_lab[:,0:128,1:3]

test_img = img_gray[:,128:256]
test_gt = img_lab[:,128:256,1:3]

## transform to B X C X W X H
train_img_ = np.expand_dims(np.expand_dims(train_img,0),0) /255.
train_gt_ = np.expand_dims(train_gt.transpose(2,0,1),0)/255.

test_img_ = np.expand_dims(np.expand_dims(test_img,0),0)/255.
test_gt_ = np.expand_dims(test_gt.transpose(2,0,1),0)/255.

net = Net()
pred_lab = np.concatenate((np.expand_dims(train_img, 2).astype(np.uint8), train_gt.astype(np.uint8)), 2)
pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_LAB2RGB)
plt.imshow(pred_rgb)
plt.show()
for i in range(100):

    y1,loss = net(train_img_,train_gt_)
    net.backward()
    net.update(0.01)
    print('epoch %d: %.4f'%(i,loss))
    pred_ab = y1[0].transpose(1,2,0)*255
    pred_lab = np.concatenate((np.expand_dims(train_img,2).astype(np.uint8),pred_ab.astype(np.uint8)),2)
    pred_rgb = cv2.cvtColor(pred_lab,cv2.COLOR_LAB2RGB)
    plt.imshow(pred_rgb)
    plt.show()