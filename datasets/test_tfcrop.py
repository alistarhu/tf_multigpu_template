# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:46:36 2019

@author: hl7356
"""

import tensorflow as tf
import matplotlib.pyplot as plt

reader = tf.WholeFileReader()
img_path = 'res1.jpg'

key, img_name_tf = reader.read(tf.train.string_input_producer([img_path.encode('utf-8')]))

image_raw_data = tf.gfile.FastGFile(img_path,'rb').read()

img_tf=tf.image.decode_jpeg(image_raw_data, channels=3, dct_method='INTEGER_ACCURATE')
#img_resize_summary = tf.summary.image('image_ys', tf.expand_dims(img_tf, 0))

#img_crop1=tf.image.crop_to_bounding_box(img_tf, 64+64, 64+64, 255,255)



#img_resize_summary = tf.summary.image('image 159-64, 159-64', tf.expand_dims(img_crop1, 0))
#
#img_crop2=tf.image.crop_to_bounding_box(img_tf, 158-64, 158-64, 255,255)
#img_resize_summary = tf.summary.image('image 158-64, 158-64', tf.expand_dims(img_crop2, 0))
#
#img_crop3=tf.image.crop_to_bounding_box(img_tf, 160-64, 160-64, 255,255)
#img_resize_summary = tf.summary.image('image 160-64, 160-64', tf.expand_dims(img_crop3, 0))

merged = tf.summary.merge_all()

with tf.Session() as sess:
#    summary_writer = tf.summary.FileWriter('F:\\test_tf_crop\\', sess.graph)
#    summary_all = sess.run(merged)
#    summary_writer.add_summary(summary_all, 0)
    #summary_writer.close()
	print(tf.shape(img_tf)[:2])
	tmp = sess.run( tf.shape(img_tf)[:2])
	print(tmp)
    #img_data_ys = tf.image.convert_image_dtype(img_tf, dtype=tf.uint8)
    #plt.figure(1)  # 图像显示
    #plt.imshow(img_tf.eval())
    
    #img_data_1 = tf.image.convert_image_dtype(img_crop1, dtype=tf.uint8)
    #plt.figure(2)
    
	#plt.imshow(img_crop1.eval())

    #plt.show()
    
# run tensorboard in shell  $ tensorboard --logdir=F:\\test_tf_crop