import torch.backends.cudnn as cudnn
from utils.utils import *
import cv2

device=""
imgsz=416
# Initialize
device = torch_utils.select_device(device)
weights="weights/last_yolov5s_results.pt"
start = time.time()
model = torch.load(weights, map_location=device)['model'].float() 
end = time.time()
print(f"Runtime to load weights is {end - start}") 

def letterbox(img, new_shape=(416,416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print(type(img))
    return img, ratio, (dw, dh)

def detect(iii):
    global model
    global device
    conf_thres=0.4
    iou_thres=0.5
    
    
    half = device.type != 'cpu'
    # model.fuse()
    model.to(device).eval()
    # dataset = LoadImages(source,iii, img_size=imgsz)


    #######
    img0 = iii # BGR
    #######
    img = letterbox(img0, new_shape=(416,416))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
   

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t0 = time.time()
    pred = model(img)[0]
    # print(pred)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t2 = time.time()
    print(f"Runtime for predict is {t2 - t0}") 
    # check empty dict or not for object
    if pred[0]==None:
        sample_output=0
    else:
        sample_output=1

    return sample_output

img="2.jpg"
iii=cv2.imread(img)
o=detect(iii)
print(f'output:{str(o)}')