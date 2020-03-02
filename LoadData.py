import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import rasterio
from rasterio.windows import Window
from pystac import (Catalog, CatalogType, Item, Asset, LabelItem, Collection)
from rasterio.features import rasterize
from shapely.geometry import box
import geopandas as gpd


from urllib.parse import urlparse

from pystac import STAC_IO


import time
import sys



def my_read_method(uri):
	parsed = urlparse(uri)
	if parsed.scheme.startswith('http'):
		return requests.get(uri).text
	else:
		return STAC_IO.default_read_text_method(uri)

STAC_IO.read_text_method = my_read_method



DataFramesTemplates = {
		"path" : [],
		"pos_x"  : [],
		"pos_y"  : [],
		"col"  : [],
		"id"   : []
	}

class HousesDataset(Dataset):
	"""Houses dataset
	
	Dataset's child class to get the references of 
	the images and labels from Houses dataset.
	
	Extends:
		Dataset
	"""

	def __init__(self, csv_file, cat_path, root_dir, transform=None):
		"""Initialisation of the dataset
		
		Create a dataset object with all required informations.
		
		Arguments:
			csv_file {str} -- [Path to the csv file with the dataset map]
			root_dir {str} -- [Root directorie of the images]
			cat_path {str} -- [Path of the catalog file]
		
		Keyword Arguments:
			transform {callable, optional} -- [Optional transform to be applied on a sample.] (default: {None})
		"""
		self.houses_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		self.train_cat = Catalog.from_file(cat_path)
		self.cols = {cols.id:cols for cols in self.train_cat.get_children()}

	def __len__(self):
		"""Give the number of images"""
		return len(self.houses_frame)

	def __getitem__(self, idx):
		"""Itemizer operator to get the image idx image
	
		Arguments:
			idx {int} -- [Index of the image to get]
		
		Returns:
			[numpy.array] -- [The idx-th image ]
		"""

		t = time.time()


		if torch.is_tensor(idx):
			idx = idx.tolist()


		# Get the image
		print("\n\n"+"-"*15)
		t = print_time(t, "Init :")


		scene_id = self.houses_frame['id'][idx]
		col = self.houses_frame['col'][idx]
		t0 = print_time(t, "Init 0 :")
		item = None
		for ii in self.cols[col].get_all_items():
			if str(ii.id )== str(scene_id):
				# print('ok')

				rasterio.open(ii.make_asset_hrefs_absolute().assets['image'].href).meta
				t0 = print_time(t0, "open :")
				item = ii
			if str(ii.id )== str(scene_id + "-labels"):
				#rasterio.open(ii.make_asset_hrefs_absolute().assets['labels'].href).meta
				gpd.read_file(ii.make_asset_hrefs_absolute().assets['labels'].href)
				t0 = print_time(t0, "gpd :")
				item_label = ii
		t0 = print_time(t0, "Boucle :")
		one_item = self.cols[col].get_item(id=item.id)
		t0 = print_time(t0, "Item :")

		rst = rasterio.open(one_item.assets['image'].href)
		t0 = print_time(t0, "rst :")

		win_sz = 1024
		x = self.houses_frame['pos_x'][idx]
		y = self.houses_frame['pos_y'][idx]

		window = Window(x, y, win_sz,win_sz) # 1024x1024 window starting at center of raster
		win_arr = rst.read(window=window)
		win_arr = np.moveaxis(win_arr,0,2)
		t0 = print_time(t0, "window :")

		win_box = box(*rasterio.windows.bounds(window, rst.meta['transform']))
		win_box_gdf = gpd.GeoDataFrame(geometry=[win_box], crs=rst.meta['crs'])
		win_box_gdf = win_box_gdf.to_crs({'init':'epsg:4326'}) # ignore FutureWarning for now, updating this arg to crs=4326 creates an error during gpd.sjoin()

		t0 = print_time(t0, "winbox :")


		t = print_time(t, "Get image :")

		print("\n\n")
		# Get labels
		tl = time.time()
		one_item_label = self.cols[col].get_item(id=item_label.id)
		#one_item_label.to_dict()
		tl = print_time(tl, "OneItem :")
		scene_labels_gdf = gpd.read_file(one_item_label.assets['labels'].href)
		print("Size of scene :",convert_bytes(sys.getsizeof(scene_labels_gdf)))
		tl = print_time(tl, "Scene :")
		gdf_chip = gpd.sjoin(scene_labels_gdf, win_box_gdf, how='inner', op='intersects')
		tl = print_time(tl, "Chip :")

		burn_val = 255
		shapes = [(geom, burn_val) for geom in gdf_chip.geometry]
		tl = print_time(tl, "Shapes :")
		if shapes == []:
			label_arr = np.zeros((win_sz,win_sz))
			self.label_in = False
		else:
			self.label_in = True
			chip_tfm = rasterio.transform.from_bounds(*win_box_gdf.bounds.values[0], win_sz, win_sz)
			label_arr = rasterize(shapes, (win_sz, win_sz), transform=chip_tfm, dtype='uint8')
		tl = print_time(tl, "arr :")


		sample = {'image': win_arr, 'labels': label_arr}
		t = print_time(t, "Get labels :")
		if self.transform:
			sample = self.transform(sample)
		t = print_time(t, "transform :")
		return sample

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, labels = sample['image'], sample['labels']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = np.transpose(image, (2, 0, 1))
		return {'image': image,#torch.from_numpy(image),
				'labels': labels}#torch.from_numpy(labels)}


class Normalize(object):
	"""Normalize an image."""

	def __call__(self, sample):
		image, labels = sample['image'], sample['labels']

		image =  image / 255
		return {'image': torch.from_numpy(image),
				'labels': torch.from_numpy(labels)}



def print_time(t, text):
	dt = time.time() - t
	print(text, np.round(dt, 3), 's')
	return time.time()


def convert_bytes(size):
   for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
	   if size < 1024.0:
		   return "%3.1f %s" % (size, x)
	   size /= 1024.0

   return size