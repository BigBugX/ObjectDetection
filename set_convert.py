"""
YOLOv3 detection COCO modification
Authorization: Will@Altizure&Haohan
Date: 2019-Apr-26
"""
## convert data in COCO into the format compatible with YOLO v3
## Authorization: Will@Altizure

from pycocotools.coco import COCO as cco
import os

cur_dir = os.getcwd()
train_dir = cur_dir + '\\train2017\\train2017\\'
val_dir = cur_dir + '\\val2017\\val2017\\'
train_ann_file = 'instances_train2017.json'
val_ann_file = 'instances_val2017.json'

def convert_bbox_format(xywh):
	# convert bbox format from [x, y, w, h], which stored in xywh[],
	# into [x_min, y_min, x_max, y_max], which outputed in xyxy[]
	# PS: for COCO, (x, y) is the upperleft point.
	xyxy = []
	x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
	xyxy.append(x) # x_min
	xyxy.append(y) # y_min
	xyxy.append(x + w) # x_max
	xyxy.append(y + h) # y_max

	return xyxy

def convert_cls_range(category_id):
	# convert the category_id obtained from COCO dataset from the
	# range (1, 91) to (0, 80) for training
	category_id = int(category_id)
	if category_id >=  1 and category_id <= 11: category_id = category_id - 1
	if category_id >= 13 and category_id <= 25: category_id = category_id - 2
	if category_id >= 27 and category_id <= 28: category_id = category_id - 3
	if category_id >= 31 and category_id <= 44: category_id = category_id - 5
	if category_id >= 46 and category_id <= 65: category_id = category_id - 6
	if category_id == 67: category_id = category_id - 7
	if category_id == 70: category_id = category_id - 9
	if category_id >= 72 and category_id <= 82: category_id = category_id - 10
	if category_id >= 84 and category_id <= 90: category_id = category_id - 11
	return category_id

train_lst = os.listdir(train_dir)
val_lst = os.listdir(val_dir)

tcoco = cco(train_ann_file)
vcoco = cco(val_ann_file)

train_ann_dict = {}
val_ann_dict = {}

# for each training img, find out all annotations of it
for item in train_lst:
	img_id = int(item[:-4])
	ann_ids = tcoco.getAnnIds(imgIds=img_id)
	ann_info = tcoco.loadAnns(ann_ids)
	# for each annotation record, obtain label(category_id) and bbox of it
	# format: label, x, y, width, height, use tmp_lst to store them
	#if (len(ann_ids) < 1):
	#	continue
	tmp_lst = []
	for ann_item in ann_info:
		# tmp_lst.append(ann_item['category_id'])
		bbox = ann_item['bbox']
		bbox = convert_bbox_format(bbox)
		for i in range(0, 4):
			tmp_lst.append(int(bbox[i]))
		tmp_lst.append(convert_cls_range(ann_item['category_id']))
	train_ann_dict[str(img_id)] = tmp_lst

# for each val img, find out all annotations of it
for item in val_lst:
	img_id = int(item[:-4])
	ann_ids = vcoco.getAnnIds(imgIds=img_id)
	ann_info = vcoco.loadAnns(ann_ids)
	# for each annotation record, obtain label(category_id) and bbox of it
	# format: label, x, y, width, height, use tmp_lst to store them
	#if (len(ann_ids) < 1):
	#	continue
	tmp_lst = []
	for ann_item in ann_info:
		# tmp_lst.append(ann_item['category_id'])
		bbox = ann_item['bbox']
		bbox = convert_bbox_format(bbox)
		for i in range(0, 4):
			tmp_lst.append(int(bbox[i]))
		tmp_lst.append(convert_cls_range(ann_item['category_id']))
	val_ann_dict[str(img_id)] = tmp_lst

with open('train.txt','w') as f:
	for i in range(0, len(train_lst)-1):
		img_path = train_dir + train_lst[i]
		img_ann = train_ann_dict[str(int(train_lst[i][:-4]))]
		#f.write(img_path + ' ' + str(img_ann) + ' ' + '\n')
		if(len(img_ann) == 0):
			continue
		f.write(img_path + ' ')
		for j in img_ann:
			f.write(str(j) + ' ')
		f.write('\n')
	img_path = train_dir + train_lst[-1]
	img_ann = train_ann_dict[str(int(train_lst[-1][:-4]))]
	#f.write(img_path + ' ' + str(img_ann) + ' ')
	f.write(img_path + ' ')
	for j in img_ann:
		f.write(str(j) + ' ')
	f.write('\n')

with open('val.txt','w') as f:
	for i in range(0, len(val_lst)-1):
		img_path = val_dir + val_lst[i]
		img_ann = val_ann_dict[str(int(val_lst[i][:-4]))]
		#f.write(img_path + ' ' + str(img_ann) + ' ' + '\n')
		if(len(img_ann) == 0):
			continue
		f.write(img_path + ' ')
		for j in img_ann:
			f.write(str(j) + ' ')
		f.write('\n')
	img_path = val_dir + val_lst[-1]
	img_ann = val_ann_dict[str(int(val_lst[-1][:-4]))]
	#f.write(img_path + ' ' + str(img_ann) + ' ')
	f.write(img_path + ' ')
	for j in img_ann:
		f.write(str(j) + ' ')
	f.write('\n')
