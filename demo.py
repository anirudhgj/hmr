"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import os
import skimage.io as io
import tensorflow as tf
import scipy.io as sio
import glob

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
from commons import skeleton_utils
from commons import vis_image as vis
from natsort import natsorted
import matplotlib.pyplot as plt

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def pose_rotate(points, theta, batch_size):
    theta = theta * np.pi / 180.0
    cos_vals = np.cos(theta)
    sin_vals = np.sin(theta)
    row_1 = np.concatenate([cos_vals, -sin_vals], axis=1)# 90 x 2
    row_2 = np.concatenate([sin_vals, cos_vals], axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    zero_size_row1x2 = np.zeros([batch_size, 1, 2])#90 x 1 x 2
    r1x2xZero = np.concatenate([row_12, zero_size_row1x2], axis=1)
    stacker = np.array([0.0, 0.0, 1.0])
    third_cols = np.reshape(np.tile(stacker, batch_size), [batch_size, 3])
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([r1x2xZero, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


def rotate_y_axis(points,theta,batch_size):
    theta = theta * np.pi / 180
    cos_vals = np.cos(theta)#90 x 1
    sin_vals = np.sin(theta)
    zero_vals = np.zeros((batch_size,1))
    ones_vals = np.ones((batch_size,1))
    row_1 = np.concatenate([cos_vals, zero_vals],axis =1)#90 x2
    row_2 = np.concatenate([zero_vals ,ones_vals],axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    temp_3 = np.stack((-sin_vals,zero_vals),axis =2)#90 x 1 x 2
    temp_32 = np.concatenate([row_12,temp_3],axis = 1)#90 x 3 x 2
    third_cols = np.concatenate([sin_vals,zero_vals,cos_vals],axis=1)#90 x 3
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([temp_32, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


def augment_pose_seq(pose_seq,z_limit=(0,360),y_limit=(-90,90)):
    pose_seq = np.expand_dims(pose_seq, axis=1)
    thetas = np.random.uniform(z_limit[0],z_limit[1], pose_seq.shape[0])
    thetas = np.stack([thetas]*pose_seq.shape[1], 1)
    k=[]
    for ct, xx in enumerate(thetas):
        k.append(pose_rotate(pose_seq[ct], np.expand_dims(thetas[ct], 1), pose_seq[ct].shape[0]))
    k = np.stack(k, 0)

    thetas = np.random.uniform(y_limit[0],y_limit[1], k.shape[0])
    thetas = np.stack([thetas]*k.shape[1], 1)
    p=[]
    for ct, xx in enumerate(thetas):
        p.append(rotate_y_axis(k[ct], np.expand_dims(thetas[ct], 1), k[ct].shape[0]))
    p = np.stack(p, 0)
    return k


def modified_mat(ske,name) :
   
    add_joint_raw = {'hips':0,
        'leftUpLeg':1,
        'rightUpLeg':2,
        'spine':3,
        'leftLeg':4,
        'rightLeg':5,
        'spine1':6,
        'leftFoot':7,
        'rightFoot':8,
        'spine2':9,
        'leftToeBase':10,
        'rightToeBase':11,
        'neck':12,
        'leftShoulder':13,
        'rightShoulder':14,
        'head':15,
        'leftArm':16,
        'rightArm':17,
        'leftForeArm':18,
        'rightForeArm':19,
        'leftHand':20,
        'rightHand':21,
        'leftHandIndex1':22,
        'rightHandIndex1':23}

    modified_joint_names = {'hips':0,
                            'neck':1,
                            'rightShoulder':2,
                            'rightForeArm':3,
                            'rightHand':4,  
                            'leftShoulder':5,
                            'leftForeArm':6,
                            'leftHand':7,
                            'head':8,
                            'rightUpLeg':9,
                            'rightLeg':10,
                            'rightFoot':11,
                            'rightToeBase':12,
                            'leftUpLeg':13,
                            'leftLeg':14,
                            'leftFoot':15,
                            'leftToeBase':16}

    if name == '2d':

        z = np.zeros((17,2))
        resized_frame = ske.copy()
        for key, value in modified_joint_names.items():
            z[value]=resized_frame[add_joint_raw[key]]

    if name == '3d':

        z = np.zeros((17,3))
        resized_frame = ske.copy()
        for key, value in modified_joint_names.items():
            z[value]=resized_frame[add_joint_raw[key]]

        one_col = z[:,[0]]
        two_col = z[:,[1]]
        three_col = z[:,[2]]
        z[:,[0]] = one_col
        z[:,[1]] = three_col
        z[:,[2]] = -two_col
        z[0]=(z[9] + z[13])/2
        z = z - z[0]
        hip_right = np.array(z[9])
        hip_left = np.array(z[13])
        pelvis_pos = (hip_right + hip_left)/2 #interpolating pelvis
        z[0] = pelvis_pos
        shoulder_left = np.array(z[5])
        shoulder_right = np.array(z[2])
        neck_pos_prior = z[1]
        neck_pos = (shoulder_left + shoulder_right)/2 #interpolating neck
        z[1] = neck_pos
        del_neck = neck_pos - neck_pos_prior
        z[8] = z[8] + del_neck #changing head_pos
        z = z - z[0] #making it root relative
        z = skeleton_utils.fit_skeleton_frame(z)
    return np.array(z)

def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    images = glob.glob('/content/gdrive/My Drive/projects/hmr/supplementary/*.png')
    print ("Prediction started")
    print (len(images))
    
    d = {}

    for n,j in enumerate(images) :
        img_arr =[]
        pose_arr = []
#         imgs = natsorted(glob.glob(j+'/*.jpg'))
        print (j)
#         for i in imgs : 
#             print (i)
#             print ('{}/{} Done'.format(n,len(images))) 
        input_img, proc_param, img = preprocess_image(j, json_path)
            # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
    #         k = '/'.join(i.split('.')[0].split('/')[-2:])

            # Theta is the 85D vector holding [camera, pose, shape]
            # where camera is 3D [s, tx, ty]
            # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
            # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

#             print (joints3d.shape)
        pose_3d = joints3d.reshape((24,3))
        po=modified_mat(pose_3d,'3d')
        poses_3d = np.squeeze(augment_pose_seq(np.expand_dims(po,0) , z_limit=(-90,-90),y_limit=(0,0)))
        fig = vis.get_figure()
        ax = vis.get_ax(poses_3d,fig)
        vis.plot_skeleton_and_scatter(poses_3d,ax)
    #         print ('/data/vcl/sid/benedict/yt_videos/17j_poses_3d/'+k+'.mat')

    #         mat_path = '/data/vcl/sid/benedict/yt_videos/17j_poses_3d/'+k+'.mat'
    #         if not os.path.exists('/'.join(mat_path.split('/')[:-1])):
    #             os.makedirs('/'.join(mat_path.split('/')[:-1]))

#             img_arr.append(img)
#             pose_arr.append(poses_3d)
    
#         mat_path = j+'/data.mat'
        if ".png" in j :
          mat_path = j.replace('.png','.mat')
          ske_plot = j.replace('.png','_ske.png')
        if ".jpg" in j :
          mat_path = j.replace('.jpg','.mat')
          ske_plot = j.replace('.jpg','_ske.jpg')
        plt.savefig(ske_plot, bbox_inches='tight',transparent=True)

        print (mat_path)
        sio.savemat(mat_path,{"pose_3d":poses_3d})
    
    
#    visualize(img, proc_param, joints[0], verts[0], cams[0])


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"]="2"

    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)