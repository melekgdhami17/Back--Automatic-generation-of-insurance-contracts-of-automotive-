from datetime import datetime
import sys
import os
from PIL import Image
import pytesseract
from pathlib import Path
import threading
from typing import Union
import cv2
import av
import numpy as np
import streamlit as st
import torch
import torch.backends.cudnn as cudnn

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import hashlib


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def detect_face(img):
    img = cv2.imread(img)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    (x, y, w, h) = faces[0][0], faces[0][1], faces[0][2], faces[0][3]

    crop = img[y - 10:y + h + 10, x - 10:x + w + 10]
    cv2.imwrite("./faces/face.png", crop)



# DB Management
import mysql.connector
import sqlite3
con = mysql.connector.connect(host="localhost",port="4001",user="root",password="",database="sentinel")
cc = con.cursor()

def add_visitor(first_name,second_name,id_number,delivery_date):
    print(first_name)
    cc.execute('INSERT IGNORE INTO visiteurs(id_cnib, nom, prenom, date_delivrance) VALUES (%s,%s,%s,%s)',(id_number,second_name,first_name,delivery_date))
    con.commit()

def add_visite(id_number,date_time,comment):
    cc.execute('SELECT * FROM visiteurs where id_cnib = "{}"'.format(id_number))
    ccResult = cc.fetchall()
    idUser = ccResult[0][0]
    print(idUser)
    cc.execute('INSERT INTO visites(id_visiteur, date_heure_entree, objet_visite) VALUES (%s,%s,%s)',(idUser,date_time, comment))
    con.commit()



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time

tic = time.time()
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def textFromTesseractOCR(croppedImage):
  text = pytesseract.image_to_string(croppedImage, lang = 'fra',config = "--psm 11 --oem 3")
  return(''.join(e for e in text if e.isalnum() or e ==  "/"))




@torch.no_grad()
def cropdetect(weights="../id_card_detection.pt",  # model.pt path(s)
               source="./tempo/input.png",  # file/dir/URL/glob, 0 for webcam
               imgsz=(640, 640),  # inference size (height, width)
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               view_img=False,  # show results
               save_txt=False,  # save results to *.txt
               save_conf=False,  # save confidences in --save-txt labels
               save_crop=True,  # save cropped prediction boxes
               nosave=False,  # do not save images/videos
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               visualize=False,  # visualize features
               update=False,  # update all models
               project=ROOT / './data',  # save results to project/name
               name='card',  # save results to project/name
               exist_ok=False,  # existing project/name ok, do not increment
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (
                        pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'im.jpg')  # im.jpg
            txt_path = str(save_dir / 'labels' / "crop") + (
                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / 'crop.jpg',
                                         BGR=True)

            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                        (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)



@torch.no_grad()
def run(weights="../best.pt" ,  # model.pt path(s)
        source="./data/card/crops/id_card/crop.jpg",  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / './data',  # save results to project/name
        name='data',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'im.jpg')  # im.jpg
            txt_path = str(save_dir / 'labels' / "crop") + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / 'crop.jpg', BGR=True)

            # Print time (inference-only)
            #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    d = {}
    for f in names:
        imgP = f"./data/data/crops/{f}/crop.jpg"
        img = Image.open(imgP)

        d[str(f)] = textFromTesseractOCR(img)
    return d

def create_usertable():
    cc.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    cc.execute('INSERT INTO agents_guerite(username,password) VALUES (%s,%s)',(username,password))
    con.commit()

def login_user(username,password):
    cc.execute('SELECT * FROM agents_guerite WHERE username =%s AND password = %s',(username, password))

    data = cc.fetchall()
    return data


def view_all_users():
    cc.execute('SELECT * FROM userstable')
    data = cc.fetchall()
    return data
def main():
    """Simple Login App"""

    st.title("OCR Naana Tech")

    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Accueil")

    elif choice == "Login":
        st.subheader("Section de connexion")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            result = login_user(username,password)
            if 1:
                agent_last_name = result[0][3]
                agent_first_name = result[0][4]

                st.success("Bienvenu {} {}".format(agent_last_name, agent_first_name))

                c = st.selectbox("Choose an option",["Open Your Webcam","Upload an ID Image"])
                if c =="Open Your Webcam":
                    def hold_cam():
                        class VideoTransformer(VideoTransformerBase):
                            frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
                            in_image: Union[np.ndarray, None]
                            out_image: Union[np.ndarray, None]

                            def __init__(self) -> None:
                                self.frame_lock = threading.Lock()
                                self.in_image = None
                                self.out_image = None

                            def transform(self, frame: av.VideoFrame) -> np.ndarray:
                                in_image = frame.to_ndarray(format="bgr24")

                                with self.frame_lock:
                                    self.in_image = in_image

                                return in_image

                        ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

                        if ctx.video_transformer:
                            c = st.columns(5)
                            if c[2].button("take picture"):
                                with ctx.video_transformer.frame_lock:
                                    in_image = ctx.video_transformer.in_image

                                if in_image is not None:
                                    cv2.imwrite("tempo/input.png", in_image)
                                    with st.spinner(text='Predict Result'):
                                        cropdetect()
                                        detect_face("./data/card/crops/id_card/crop.jpg")
                                        d = run()
                                        c = st.columns(5)
                                        b = c[2].button("save data")
                                        c = st.columns(5)
                                        c[2].subheader("Results:")

                                        with st.expander("ID Number"):
                                            id_number = str(st.text(d["id"]))

                                        with st.expander("FIRST NAME"):
                                            first_name = str(st.text(d["first_name"]))


                                        with st.expander("LAST NAME"):
                                            second_name = str(st.text(d["last_name"]))

                                        with st.expander("Delivery Date"):
                                            delivery_date = str(st.text(d["delivery"]))

                                        now = datetime.now()
                                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                                        with st.expander("Date and Time"):
                                            date_time = str(st.text(dt_string))
                                        with st.expander("Add Comment"):
                                            comment = str(st.text_area(""))
                                        if b:
                                            add_visitor(d["first_name"], d["second_name"], d["id_number"], d["delivery"])
                                            add_visite(d["id"],date_time,comment)

                                        st.sidebar.title("Detected ID Card: ")
                                        st.sidebar.image("./data/card/crops/id_card/crop.jpg")
                                else:
                                    st.warning("No frames available yet.")
                    hold_cam()



                else:
                    img_file_buffer = st.file_uploader("Upload your ID Card", type=["png", "jpg", "jpeg"])
                    if img_file_buffer is not None:
                        file_details = {"FileName": img_file_buffer.name, "FileType": img_file_buffer.type}

                        with open(os.path.join("tempo", "input.png"), "wb") as f:
                            f.write(img_file_buffer.getbuffer())

                        cropdetect()
                        detect_face("./data/card/crops/id_card/crop.jpg")
                        d = run()

                        st.success("Saved File")
                        st.markdown("---")
                        with st.spinner(text='Predict Result'):
                            c = st.columns(5)
                            b = c[2].button("save data")
                            c = st.columns(5)
                            c[2].subheader("Results:")

                            with st.expander("ID Number"):
                                id_number = str(st.text(d["id"]))

                            with st.expander("FIRST NAME"):
                                first_name = str(st.text(d["first_name"]))

                            with st.expander("LAST NAME"):
                                second_name = str(st.text(d["last_name"]))

                            with st.expander("Delivery Date"):
                                delivery_date = str(st.text(d["delivery"]))

                            now = datetime.now()
                            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                            with st.expander("Date and Time"):
                                date_time = str(st.text(dt_string))
                            with st.expander("Add Comment"):
                                comment = str(st.text_area(""))
                            if b:
                                add_visitor(d["first_name"], d["second_name"], d["id_number"], d["delivery"])
                                add_visite(d["id"], date_time, comment)

                            st.sidebar.title("Detected ID Card: ")
                            st.sidebar.image("./data/card/crops/id_card/crop.jpg")
                            if b:
                                add_visitor(d["first_name"], d["second_name"], d["id"], d["delivery"])
                                add_visite(d["id"], date_time, comment)

                            st.sidebar.title("Detected ID Card: ")
                            st.sidebar.image("./tempo/input.png")

            else:
                st.warning("Incorrect Username/Password")





    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()