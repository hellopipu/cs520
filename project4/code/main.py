import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from skimage.measure import compare_psnr,compare_ssim

np.random.seed(66)

def convert_rgb_to_gray(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    return 0.21*r+0.72*g+0.07*b
def cal_rgb_dist(c1,c2):
    # print('c1 ',c1)
    # print('c2 ', c2)
    return (2.*(c1[0]-c2[0])**2+4.*(c1[1]-c2[1])**2+3.*(c1[2]-c2[2])**2)**0.5
def find_closest_cluster(rgb,cluster):
    l=[]
    # print('cluster: ',cluster)
    for c in cluster:
        # print(rgb,c)
        l.append(cal_rgb_dist(rgb,c))
    # print('l: ',l)
    return l.index(min(l))


# @timer(True)
def find_representative_colors(img,k,repeat):
    w,h,_ = img.shape
    img_ = img.reshape(w*h,3)
    cluster = [1.*img_[i] for i in np.random.randint(0,w*h,k)]
    for _ in tqdm(range(repeat)):
        l = [[] for _ in range(k)]
        ## find each point belong to which cluster
        for i in range(w*h):
            l[find_closest_cluster(img_[i],cluster)].append(i)
        ## update clusters
        for i in range(k):
            cluster[i] = np.mean([ img_[j] for j in l[i]],0)
        # print(cluster)
    return cluster
def recolor_by_representative(img,cluster):
    w,h,_ = img.shape
    img_recolor = np.zeros(img.shape,np.uint8)
    for i in range(w):
        for j in range(h):
            img_recolor[i,j]= cluster[find_closest_cluster(img[i,j],cluster)]
            # print(img_recolor[i,j])
    return img_recolor
def get_patch(img,i,j):

    w,h = img.shape
    if 0<i<w-1 and 0<j<h-1:
        return img[i-1:i+2,j-1:j+2]
    else:
        a = np.zeros((3,3))
        for m in [-1,0,1]:
            for n in [-1,0,1]:
                if 0 <= i + m < w and 0 <= j + n < h:
                    a[1+m,1+n] = img[i+m,j+n]
                else:
                    a[1 + m, 1 + n] = img[i , j ]
        return a


def cal_sim_score(patch1,patch2):
    # print(patch1,patch2)
    return np.sum(np.square(patch1-patch2))
    # return 1.* sim
    # return 1
# @timer(True)
def find_similar_patch(patch_test,img_train):
    w,h = img_train.shape
    score=[np.float('inf') for _ in range(6)]
    pos = [[] for _ in range(6)]
    tt = 0
    tt2 = 0
    for i in range(w):
        for j in range(h):
            t0=time.time()
            pp = get_patch(img_train, i, j)
            tt += time.time() - t0
            ss = cal_sim_score(pp,patch_test)
            tt2+=time.time()-t0
            if ss<max(score):
                ind = score.index(max(score))
                score[ind] = ss
                pos[ind] = [i,j]
    # print(tt,' s')
    # print(tt2,' s2')
    return score,pos



def recolor_by_patch(img_test,img_recolor,img_train,cluster):
    w,h = img_test.shape
    img_result = np.zeros((w,h,3),np.uint8)
    # print(cluster)
    tt=0
    tt1=0
    tt2=0
    for i in tqdm(range(w)):
        for j in range(h):
            t0=time.time()
            patch_test = get_patch(img_test,i,j)
            t1=time.time()
            score,pos = find_similar_patch(patch_test,img_train)
            t2=time.time()
            ind = score.index(min(score))
            img_result[i,j] =  img_recolor[pos[ind][0],pos[ind][1]] #cluster[find_closest_cluster(img_train_color[pos[ind][0],pos[ind][1]],cluster)]
            t3=time.time()
            tt+=t1-t0
            tt1+=t2-t1
            tt2+=t3-t2
        # print(tt,tt1,tt2)
    return img_result





repeat = 30
## you can modify the img size to speed the running process
sz = 256
k=5
img_bgr = cv2.imread('wallhaven-vmm37p.jpg')
img_bgr = cv2.resize(img_bgr,(sz,sz))
img_rgb = img_bgr[:,:,::-1]
img_gray = convert_rgb_to_gray(img_rgb)
w,h,_ = img_rgb.shape
print('1. k-means to find the representative color and recolor...')
c = find_representative_colors(img_rgb[:,0:int(sz/2)],k,repeat)
img_recolor = recolor_by_representative(img_rgb[:,0:int(sz/2)],c)

# plt.subplot(321)
plt.imshow(img_rgb)
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_rgb.png'%sz,dpi=300)


# plt.subplot(322)
plt.imshow(img_gray,'gray')
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_gray.png'%sz,dpi=300)

# print(c)
# plt.subplot(323)
plt.imshow(np.array(c).reshape((1,k,3)).astype(np.uint8))
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_color.png'%sz,dpi=300)
# print(np.array(c).reshape((1,5,3)))


# plt.subplot(324)
plt.imshow(img_recolor)
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_recolor.png'%sz,dpi=300)
# print(np.unique(img_recolor))
print('2. find similar patch and recolor...')
img_recolor_by_patch = recolor_by_patch(img_gray[:,int(sz/2):sz],img_recolor,img_gray[:,0:int(sz/2)],c)
# plt.subplot(325)
plt.imshow(img_recolor_by_patch)
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_recolor_by_patch.png'%sz,dpi=300)

# plt.subplot(326)
img_final = np.hstack((img_recolor,img_recolor_by_patch))
plt.imshow(img_final)
plt.xticks([]),plt.yticks([])
plt.savefig('%d/img_final.png'%sz,dpi=300)

print('3. calculate quantitative results...')
PSNR = compare_psnr(img_rgb[:,int(sz/2):sz],img_final[:,int(sz/2):sz],255)
SSIM = compare_ssim(img_rgb[:,int(sz/2):sz],img_final[:,int(sz/2):sz],multichannel=True)
print('psnr: ',PSNR)
print('ssim: ',SSIM)