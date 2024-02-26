"""
    2024南开大学计算机视觉实验室实习笔试
    
    基于 SAM (Segment Anything Model) 的交互式图像分割 napari图形界面
        by 郝家诚
    2024.01.12
    功能：
        1. Point-Prompt 分割
        2. 全图分割
        3. 并列展示分割结果

    1.18更新：
        功能3实现
        bug:先point-prompt 再full segmentation 会出现bug
    
    环境: pytorch
    1.22更新：
        import_image 会更新序号为0的图片序列，加入mask序列后出现bug

    1.23:
        第5张图片以后（sam_gap），就不进行point-prompt分割了，只进行全图分割
    
    v6
    1.24:
    s:
        1. 可以对全局分割&跟踪
        2. 可以加入point,对单个物体分割跟踪
        3. 对第一张图可refine
    w:
        1. 只能对image_sequence[0]进行refine
        2. 不能跟踪多个物体

    v7
    1.25:
        新增 function add_object, 可以跟踪多个物体
"""

from datetime import datetime
# from PyQt5.QtCore import Qt
import napari
import imageio.v2 as imageio
import skimage.io as io
# import torch
# import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import QSlider, QHBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QInputDialog
import cv2
import os
import random
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from typing import Any, Dict, List
import json
import matplotlib.pyplot as plt
import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker

def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                    origin_frame=origin_frame, 
                                                    coords=np.array(prompt["points_coord"]),
                                                    modes=np.array(prompt["points_mode"]),
                                                    multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame, predicted_mask


def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    
    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }

    return prompt



class SliderDock(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.mask_sequence = []
        self.output = './'
        self.updated_data = []
        self.updated_Negdata = []
        self.exist_mask = False
        self.masksq = False   #   mask序列是否存在，若不存在，则加入mask序列，若存在，则更新mask序列
        layout = QHBoxLayout()
        self.slider = QSlider()
        self.slider.setOrientation(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(image_sequence) - 1)
        self.slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.slider)
        # 增加文本框，实时显示当前图片序列索引
        self.textbox2 = QLineEdit()
        self.textbox2.setText("0")
        self.textbox2.setReadOnly(True)
        # 设置长度
        self.textbox2.setFixedWidth(40)
        layout.addWidget(self.textbox2)

        self.tracker = SegTracker(segtracker_args,sam_args,aot_args)
        self.tracker.restart_tracker()
        
        import_button = QPushButton("Import Images")
        import_button.clicked.connect(self.import_images)
        layout.addWidget(import_button)

        # outfolder = QPushButton("Output Folder")
        # outfolder.clicked.connect(self.output_folder)
        # layout.addWidget(outfolder)

        addpoints = QPushButton("Add Points")
        addpoints.clicked.connect(self.add_points)
        layout.addWidget(addpoints)

        addNegpoints = QPushButton("Add NegPoints")
        addNegpoints.clicked.connect(self.add_Negpoints)
        layout.addWidget(addNegpoints)

        addobj = QPushButton("Add Object")
        addobj.clicked.connect(self.add_object)
        layout.addWidget(addobj)

        seleobj = QPushButton("Select Object")
        seleobj.clicked.connect(self.select_object)
        layout.addWidget(seleobj)

        seg_button = QPushButton("Segment")
        seg_button.clicked.connect(self.segment_image)
        layout.addWidget(seg_button)

        track_button = QPushButton("Track")
        track_button.clicked.connect(self.track_image)
        layout.addWidget(track_button)

        self.textbox = QLineEdit()
        self.textbox.setText("当前操作对象id: "+str(self.tracker.curr_idx))
        self.textbox.setReadOnly(True)
        layout.addWidget(self.textbox)


        self.setLayout(layout)

    def add_Negpoints(self):
        Negpoints_layer = viewer.add_points(size=5, face_color='blue', edge_color='green', name='NegPoints')
        viewer.layers['NegPoints'].mode = 'add'
        self.Negpoints_data = viewer.layers['NegPoints'].data
        def Negpoints_callback(event):
        # 获取更新后的点数据
            self.updated_Negdata = Negpoints_layer.data
            # self.updated_Negdata[:, [0, 1]] = self.updated_Negdata[:, [1, 0]]
        # 打印最新的点坐标
            print("更新后negative点的坐标:", self.updated_Negdata)
        
        # 点数据变化的事件回调
        Negpoints_layer.events.data.connect(Negpoints_callback)

    def add_points(self):
        points_layer = viewer.add_points(size=5, face_color='red', edge_color='green', name='Points')
        viewer.layers['Points'].mode = 'add'
        self.points_data = viewer.layers['Points'].data
        def points_callback(event):
        # 获取更新后的点数据
            self.updated_data = points_layer.data
            # self.updated_data[:, [0, 1]] = self.updated_data[:, [1, 0]]
        # 打印最新的点坐标
            print("更新后positive点的坐标:", self.updated_data)
        
        # 点数据变化的事件回调
        points_layer.events.data.connect(points_callback)

    def select_object(self):
        min_value = 1  # 最小值
        max_value = self.tracker.get_obj_num()  # 最大值

        # 弹出对话框，允许用户输入数字
        number, ok = QInputDialog.getInt(None, 'Input Number', 'Enter a number:',
                                            min=min_value, max=max_value, step=1)

        if ok:
            self.tracker.curr_idx = number
            self.textbox.setText("当前操作对象id: "+str(self.tracker.curr_idx))

    # def output_folder(self):
    #     file_dialog = QFileDialog()
    #     file_dialog.setFileMode(QFileDialog.Directory)
    #     out_file_path = file_dialog.getExistingDirectory(self, "Select Folder")
    #     if out_file_path:
    #         self.output = out_file_path
    #         self.textbox.setText("Output folder: "+ self.output)

    def update_image(self, value):
        self.viewer.layers['image'].data = image_sequence[value]
        if self.exist_mask == True:
            self.viewer.layers['mask'].data = self.mask_sequence[value]   ######序号要改成名字
        self.textbox2.setText(str(value))
        
    def import_images(self):
        folder_dialog = QFileDialog()
        
        # 获取选定的文件夹路径
        folder_path = folder_dialog.getExistingDirectory(self, "Select Folder")

        img_path = folder_path
        
        contents = os.listdir(img_path)
        image_files = [os.path.join(img_path, file) for file in contents if file.endswith('.tif')]
        image_sequence.extend([np.array(Image.open(file)) for file in image_files])
        for i, image in enumerate(image_sequence):
            if len(image.shape) == 2:
                image_sequence[i] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
        viewer.add_image(image_sequence[0], name='image')

        #     print(new_images[0].shape)
        #     print(image_sequence[0].shape)
        #     image_sequence.extend(new_images)
        self.slider.setMaximum(len(image_sequence) - 1)
        self.viewer.layers[0].data = image_sequence[0]
        self.viewer.layers[0].refresh()
        
        ################   读取视频，抽帧  ################
        # cap = cv2.VideoCapture(io_args['input_video'])
        # with torch.cuda.amp.autocast():
        #     while cap.isOpened():
        #         ret, frame = cap.read()    ############ 读取视频帧  ############
        #         if not ret:
        #             break
        #         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    ##########  将视频帧转为RGB格式  ##########
        #         image_sequence.append(frame)
        #         if len(image_sequence) == 1:
        #             viewer.add_image(image_sequence[0], name='image')
        #         self.slider.setMaximum(len(image_sequence) - 1)
        #     self.viewer.layers['image'].data = image_sequence[0]
        #     self.viewer.layers['image'].refresh()
        #     cap.release()
    def add_object(self):
        prev_mask = self.tracker.first_frame_mask
        self.tracker.update_origin_merged_mask(prev_mask) 
        self.tracker.curr_idx += 1
        self.textbox.setText("当前操作对象id: "+str(self.tracker.curr_idx))

    
    def segment_image(self):
        click_stack = [[],[]]
        print("当前对象id: ", self.tracker.curr_idx)
        print("跟踪对象总数量: ", self.tracker.get_obj_num() + 1)


        
        ############ 输出路径 ############
        output_dir = io_args['output_mask_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ############ 定义模型 ############
        torch.cuda.empty_cache()
        gc.collect()
        
        
        ################ 坐标转换
        if len(self.updated_data) >= 1:
            
            for i in range(len(self.updated_data)):
                pos_coord = np.ndarray.astype(self.updated_data[i], np.int32)
                
                point = {"coord": pos_coord[::-1], "mode": 1}
                print(point)
                click_prompt = get_click_prompt(click_stack, point)    #   坐标进入prompt

            if len(self.updated_Negdata) != 0:
                
                for i in range(len(self.updated_Negdata)):
                    neg_coord = np.ndarray.astype(self.updated_Negdata[i], np.int32)
                    
                    point = {"coord": neg_coord[::-1], "mode": 0}  #    flatten()将数组变为一维
                    click_prompt = get_click_prompt(click_stack, point)
            
            # Refine acc to prompt
            masked_frame, pred_mask = seg_acc_click(self.tracker, click_prompt, self.viewer.layers['image'].data)    
            if len(self.mask_sequence) == 0 and self.exist_mask == False:
                self.mask_sequence.append(masked_frame)
                
                viewer.add_image(self.mask_sequence[0], name='mask')
                self.exist_mask = True
            elif len(self.mask_sequence) == 0 and self.exist_mask == True:
                self.mask_sequence.append(masked_frame)
            else:
                self.mask_sequence[0] = masked_frame
            self.viewer.layers['mask'].data = masked_frame


    def track_image(self):
        # self.viewer.layers[1].data = mask_sequence[self.slider.value()]
        sam_gap = segtracker_args['sam_gap']
        masked_pred_list = []
        self.exist_mask == True
        frame_idx = 0
        # image_index = self.slider.value()
        # image = image_sequence[image_index]
        ############  导入视频  ############
        if len(self.updated_data) >= 1:
            with torch.cuda.amp.autocast():
            # while cap.isOpened():
            #     ret, frame = cap.read()    ############ 读取视频帧  ############
            #     if not ret:
            #         break
                for frame in image_sequence:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    ##########  将视频帧转为RGB格式  ##########

                    if frame_idx == 0:
                        pred_mask = self.tracker.first_frame_mask
                        torch.cuda.empty_cache()
                        gc.collect()
                    elif (frame_idx % sam_gap) == 0:
                        seg_mask = self.tracker.seg(frame)
                        torch.cuda.empty_cache()
                        gc.collect()
                        track_mask = self.tracker.track(frame)
                        # find new objects, and update tracker with new objects
                        # new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)
                        # save_prediction(new_obj_mask, output_mask_dir, str(frame_idx+frame_num).zfill(5) + '_new.png')
                        pred_mask = track_mask
                        # segtracker.restart_tracker()
                        self.tracker.add_reference(frame, pred_mask)
                    else:
                        pred_mask = self.tracker.track(frame,update_memory=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    masked_frame = draw_mask(frame,pred_mask)
                    if self.masksq == False:
                        self.mask_sequence.append(masked_frame)
                    else:
                        self.mask_sequence[frame_idx] = masked_frame
                            # save_prediction(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
                    # formatted_idx = str(frame_idx).zfill(3)
                    # image.convert('I;16').save(output_path + f'mask{formatted_idx}.tif')

                    
                    self.viewer.layers['mask'].data = masked_frame
                    # masked_frame = draw_mask(frame,pred_mask)
                    # masked_pred_list.append(masked_frame)
                    # plt.imshow(masked_frame)
                    # plt.show() 

                    
                    
                    print("processed frame {}, obj_num {}".format(frame_idx,self.tracker.get_obj_num()),end='\r')
                    frame_idx += 1
                
                self.masksq = True

        else:
            with torch.cuda.amp.autocast():
                # while cap.isOpened():
                #     ret, frame = cap.read()    ############ 读取视频帧  ############
                #     if not ret:
                #         break
                for frame in image_sequence:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    ##########  将视频帧转为RGB格式  ##########
                    ##########   Tracking   ##########
                    if frame_idx == 0:
                        pred_mask = self.tracker.seg(frame)
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.tracker.add_reference(frame, pred_mask)
                    elif (frame_idx % sam_gap) == 0:     ###############   分割关键帧来跟踪   ###############
                        seg_mask = self.tracker.seg(frame)
                        torch.cuda.empty_cache()
                        gc.collect()
                        track_mask = self.tracker.track(frame)
                        # find new objects, and update tracker with new objects
                        new_obj_mask = self.tracker.find_new_objs(track_mask,seg_mask)
                        # save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                        pred_mask = track_mask + new_obj_mask
                        self.tracker.restart_tracker()
                        self.tracker.add_reference(frame, pred_mask)
                    else:
                        pred_mask = self.tracker.track(frame,update_memory=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("pred_mask:", pred_mask.shape, "pred_obj", np.unique(pred_mask))
                    image = Image.fromarray(pred_mask.astype(np.uint8))
                    # 保存为.tif文件
                    formatted_idx = str(frame_idx).zfill(3)
                    # image.convert('I;16').save(output_path + f'mask{formatted_idx}.tif')
                    # save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')
                    masked_frame = draw_mask(frame,pred_mask)
                    self.mask_sequence.append(masked_frame)
                    if len(self.mask_sequence) == 1:
                        viewer.add_image(self.mask_sequence[0], name='mask')
                    
                    self.viewer.layers['mask'].data = masked_frame
                    # masked_frame = draw_mask(frame,pred_mask)
                    # masked_pred_list.append(masked_frame)
                    # plt.imshow(masked_frame)
                    # plt.show() 
                    
                    
                    print("processed frame {}, obj_num {}".format(frame_idx,self.tracker.get_obj_num()),end='\r')
                    frame_idx += 1
                
                print('\nfinished')




video_name = 'cell'
io_args = {
    'input_video': f'./assets/{video_name}.mp4',
    'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
    'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization
}



segtracker_args = {
    'sam_gap': 5, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}
image_sequence = []

click_stack = [[],[]]
output_path = "F:/sth/23Fall/NKCVintern/output/02/02_RES/"
viewer = napari.Viewer()
slider_dock = SliderDock(viewer)
viewer.window.add_dock_widget(slider_dock, area='bottom')
# viewer.add_image(image_sequence[0])
# viewer.add_image(mask_sequence[0], visible=False)
napari.run()

