import numpy as np
import torch
import win32con
import win32gui
import win32ui
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


def draw(img0, boxs, re_x, re_y, thickness=2):
    """
    绘制
    """
    if len(boxs):
        for i, det in enumerate(boxs):
            _, x_center, y_center, width, height = det
            x_center, width = re_x * float(x_center), re_x * float(width)
            y_center, height = re_y * float(y_center), re_y * float(height)
            top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
            bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
            color = (0, 0, 255)  # RGB
            cv2.rectangle(img0, top_left, bottom_right, color, thickness=thickness)
            cv2.putText(img0, str(_), top_left, 0, 1, color, thickness, cv2.LINE_8, False)


def grab_screen_win32(region):
    """
    屏幕截图
    """
    hwin = win32gui.GetDesktopWindow()
    left, top, x2, y2 = region
    width = x2 - left + 1
    height = y2 - top + 1

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def load_model(weights, data):
    device = select_device('')
    return DetectMultiBackend(weights, device=device, data=data, dnn=False, fp16=False)


@smart_inference_mode()
def get_label_data(
        model,  # 模块
        img0,  # 原图像
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
):
    # 处理图片
    im = letterbox(img0, 640, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # Run inference
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=augment, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # 结果转换为 label format 保存到 boxs
    boxs = []
    for i, det in enumerate(pred):  # per image
        im0 = img0.copy()
        s = ' '
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            box = ('%g ' * len(line)).rstrip() % line
            box = box.split(' ')
            boxs.append(box)
    return boxs


def run():
    winname = "realtime_win"
    region = (0, 0, 1920, 1080)
    re_x, re_y = (1920, 1080)
    weights = "D:\Programming\Projects\Pycharm\yolov5\yolov5s.pt"
    data = None

    model = load_model(weights, data)
    while True:
        img0 = grab_screen_win32(region=region)
        label_data = get_label_data(model, img0)
        draw(img0, label_data, re_x, re_y)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow(winname)
            break

        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, re_x // 2, re_y // 2)
        cv2.imshow(winname, img0)
        HWND = win32gui.FindWindow(None, winname)
        win32gui.SetWindowPos(HWND, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)


if __name__ == '__main__':
    run()
