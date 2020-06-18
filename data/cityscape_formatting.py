import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Cityscape dataset formatting script')
parser.add_argument('--data_path', default=None, action='store', type=str, help='Root directory of the Cityscape dataset')

args = parser.parse_args()

def check_folders(path):
	for f in ['train', 'val', 'test']:
		assert os.path.exists(os.path.join(path, f)), "Folder "+str(f)+" does not exist !"


def restructure_folders(src_path, dest_path, suffix, write_ids=False):
	image_id_folder = os.path.join(dest_path, 'ImageSets')
	if not os.path.exists(image_id_folder):
		os.mkdir(image_id_folder, 0o777)
		print("ImageSets folder created")
	for f in ['train', 'val', 'test']:
		curr_dir = os.path.join(src_path, f)
		new_dir = os.path.join(dest_path, f, suffix)
		if not os.path.exists(new_dir):
			os.mkdir(new_dir, 0o777)
			print(new_dir, "created.")
		if len(os.listdir(curr_dir)) == 0:
			print("Folder", curr_dir, "is empty!")
		else:
			if write_ids:
				idFile = open(os.path.join(image_id_folder, str(f)+".txt"), 'w+')
			for fldr in os.listdir(curr_dir):
				curr_folder =  os.path.join(curr_dir, fldr)
				if curr_folder == new_dir:
					continue
				for fl in os.listdir(curr_folder):
					shutil.move(os.path.join(curr_folder, fl), new_dir)
					if write_ids:
							idFile.write('{}\n'.format(fl.split('.')[0]))
				os.rmdir(curr_folder)
			if write_ids:
				idFile.close()
				print(str(f)+".txt written.")


print("! CONTINUING WILL AFFECT THE FOLDER STRUCTURE !")
x = input("continue ? (y/n) : ")
if x == 'y':
	assert args.data_path != None, "Data path not provided !"
	assert os.path.exists(args.data_path), "Invalid Data path provided !"
	image_folder = os.path.join(args.data_path, 'leftImg8bit')
	anno_folder = os.path.join(args.data_path, 'gtFine')
	assert os.path.exists(image_folder), "Image folder does not exist !"
	assert os.path.exists(anno_folder), "Annotations folder does not exist !"
	check_folders(image_folder)
	check_folders(anno_folder)
	restructure_folders(image_folder, image_folder, 'image_2', write_ids=True)
	restructure_folders(anno_folder, image_folder, 'instance')
	print("Restructuring folders done !")

