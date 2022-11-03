import torchvision.transforms as transforms
import numpy as np
#import cv2

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

vit_train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

# class CropFace(object):
#     # val:  mean    std     median
#     # cx:   187.77  22.20   188.0
#     # cy:   250.18  43.63   251.0
#     # w, h: 192.19  54.68   196.0

#     def __init__(self,
#         x_stat=(188,22), y_stat=(250,44), 
#         w_stat=(192,55), h_stat=(192,55),
#         pic_wh=(512,384)) -> None:

#         self.f_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +\
#             "haarcascade_frontalface_default.xml")

#         self.x_mean, self.x_std = x_stat
#         self.y_mean, self.y_std = y_stat
#         self.w_mean, self.w_std = w_stat
#         self.h_mean, self.h_std = h_stat
#         self.pic_height, self.pic_width = pic_wh
#         self.pic_ratio = 1.5 # or use self.pic_height / self.pic_width
    
#     def _transform_rects(self, rects):
#         transformed_rects = []
#         for (lx, ly, w, h) in rects:
#             hw = w // 2
#             hh = h // 2
#             cx = lx + hw
#             cy = ly + hh
#             transformed_rects.append((cx, cy, hw, hh))
#         return transformed_rects

#     def _get_dist(self, point, center=(None,None)):
#         x1, y1 = point
#         x2, y2 = center
#         if center == (None, None):
#             x2, y2 = self.x_mean, self.y_mean
#         return np.sqrt((x2-x1)**2 + (y2-y1)**2).item()

#     def _sort_by_size(self, rects):
#         key = lambda rect: rect[2] * rect[3]
#         rects_s = sorted(rects, key=key, reverse=True)
#         return rects_s

#     def _is_in_area(self, point, sigma=1):
#         x, y = point
#         cx, cy = self.x_mean, self.y_mean
#         h_radius, v_radius = sigma*self.x_std, sigma*self.y_std
#         boundary = ((cx-x)**2 / h_radius**2) + ((cy-y)**2 / v_radius**2)
#         return boundary <= 1.
    
#     def _get_face_pos(self, pic, is_BGR=True):
#         center_x = self.x_mean
#         center_y = self.y_mean
#         half_width = int((self.w_mean / 2) + self.w_std)
#         half_height = int((self.h_mean / 2) + self.h_std)
        
#         pic_rgb = pic
#         if is_BGR:
#             pic_rgb = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
#         pic_gray = cv2.cvtColor(pic_rgb, cv2.COLOR_RGB2GRAY)

#         face_cnddt = self.f_cascade.detectMultiScale(
#             pic_gray,
#             scaleFactor=1.01,
#             minNeighbors=1,
#             minSize=(120, 120)
#         )

#         if type(face_cnddt) == np.ndarray:
#             f_rects = self._transform_rects(face_cnddt)
#             f_rects = [(x,y,w,h) for (x,y,w,h) in f_rects if self._is_in_area((x,y))]

#             if len(f_rects) != 0:
#                 f_rects = self._sort_by_size(f_rects)
#                 center_x, center_y, half_width, half_height = f_rects[0]
 
#         return center_x, center_y, half_width, half_height

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"

#     def __call__(self, pic, is_BGR=True):
#         seg = lambda x, MAX: 0 if x < 0 else int(x) if x < MAX else MAX
#         H = self.pic_height
#         W = self.pic_width
#         x, y, w, h = self._get_face_pos(pic, is_BGR=is_BGR)
#         h = int(self.pic_ratio * w)
#         return pic[seg(y-1.1*h,H):seg(y+0.9*h,H),
#             seg(x-w,W):seg(x+w,W)]

ColorJitter =  transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor()
])

RandomAffine = transforms.Compose([
    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.ToTensor()
])

RandomHorizontalFlip = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor()
])

RandomRotation = transforms.Compose([
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.ToTensor()
])

RandomVerticalFlip = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
])

GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor()
])

RandomPerspective = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.ToTensor()
])

RandomAdjustSharpness = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor()
])

RandomAutocontrast = transforms.Compose([
    transforms.RandomAutocontrast(),
    transforms.ToTensor()
])