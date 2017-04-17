# run it like: img.jpg mixed4c 20 1.5 8 1.4 False
#python main.py img layer iters step octaves scale printLayers

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import time
import datetime

frameNum = 1;
model_fn = 'tensorflow_inception_graph.pb'
#model_fn = 'output_graph.pb'

img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)

op = sess.graph.get_operations()

tf.import_graph_def(graph_def, {'input': t_preprocessed})
#tf.import_graph_def(graph_def)
def getTs():
	ts = time.time()
	return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

def showarray(a , fmt, layer, iter_n, step, octave_n, channel, rands):
	a = np.uint8(np.clip( a, 0, 1)*255)
	f = BytesIO()
	global frameNum
	im = PIL.Image.fromarray(a)#.save(f, fmt)
	im.save('savedImages2/' + str(frameNum).rjust(5, '0') + '.jpg') 
	frameNum = frameNum + 1
	#im.save('savedImages2/'+ layer + '_' + iter_n + '_' + step + '_' + octave_n + '_' + channel + '_'+ getTs() + '.jpg')
	#im.save('savedImages/'+ layer + '_' + iter_n + '_' + step + '_' + octave_n + '_' + str(rands[0]) + '_' + str(rands[1]) + '_' + str(rands[2]) + '_'+ getTs() + '.jpg')
	im.show()

def T(layer):
	#return graph.get_tensor_by_name("%s:0"%layer)
	return graph.get_tensor_by_name('import/'+ layer + ':0')

def tffunc(*argtypes):
	placeholders = list(map(tf.placeholder, argtypes))
	def wrap(f):
		out = f(*placeholders)
		def wrapper(*args, **kw):
			return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
		return wrapper
	return wrap

def resize(img, size):
	img = tf.expand_dims(img, 0)
	return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
	sz = tile_size
	h, w = img.shape[:2]
	sx, sy = np.random.randint(sz, size=2)
	img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
	grad = np.zeros_like(img)
	#print(grad.shape, h, w, img_shift.shape)
	#for node in sess.graph.as_graph_def().node:
	#	print node

	for y in range(0, max(h-sz//2, sz), sz):
		for x in range(0, max(w-sz//2, sz),sz):
			sub = img_shift[y:y+sz, x:x+sz]
			g = sess.run(t_grad, {t_input:sub})
			#g = sess.run(graph.get_tensor_by_name('import/mixed/tower_2/conv:0'), {'import/Mul:0':np.random.uniform(size=(1,299,299,3))})
			#g = sess.run(graph.get_tensor_by_name('import/mixed/tower_2/conv:0'), {'import/Mul:0':sub})
			#print(g.shape)
			grad[y:y+sz, x:x+sz] = g
	return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(layer='mixed4c', img0=img_noise,
		iter_n = 10, step=1.15, octave_n=4, octave_scale=1.4, printLayers=False, channel=0):
	print('layer = ' + str(layer) + ', iter = ' + str(iter_n) + ", step = " + str(step) + ", octave_n = " + str(octave_n))

	if printLayers == 'True':
		op = sess.graph.get_operations()
		for m in op:
			print(m.values())

	rands = np.random.randint(136, size=3)
	# diff ways of sampling network
	#t_obj = tf.square( T(layer)[:,:,:,rands[0]]+T(layer)[:,:,:,rands[1]]+T(layer)[:,:,:,rands[2]])
	t_obj = tf.square( T(layer)[:,:,:,channel]) 
	#t_obj = T(layer)
	#t_obj = tf.square(T(layer))

	t_score = tf.reduce_mean(t_obj)
	t_grad  = tf.gradients(t_score, t_input)[0]
	img = img0
	octaves = []
	for i in range(octave_n -1):
		hw = img.shape[:2]
		lo = resize(img, np.int32(np.float32(hw)/octave_scale))
		hi = img-resize(lo, hw)
		img = lo
		octaves.append(hi)

	for octave in range(octave_n):
		if octave > 0:
			hi = octaves[-octave]
			img = resize(img, hi.shape[:2])+hi
		for i in range(iter_n):
			g = calc_grad_tiled(img, t_grad)
			img += g*(step / (np.abs(g).mean()+1e-7))
	showarray(img / 255.0,'jpeg', layer,str(iter_n),str(step),str(octave_n), str(channel),rands )



for i in range(1,450):
	t = '../mustardFrames/crop/'+str(i).rjust(5,'0')+'.jpg'
	img0 = PIL.Image.open(t);
	if i > 1 :
		img1 = PIL.Image.open('savedImages2/'+str(i-1).rjust(5, '0')+'.jpg')
		img0 = PIL.Image.blend(img0, img1, 0.25)
	#print(t)
	#img0 = PIL.Image.open(str(sys.argv[1]))
	img0 = np.float32(img0)
	render_deepdream(str(sys.argv[2]), img0, int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), sys.argv[7], 25 )
