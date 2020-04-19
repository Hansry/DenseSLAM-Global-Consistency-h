import glob
import cv2
import os
import math
from PIL import Image
import numpy as np
import math
import torch

lg_e_10 = math.log(10)
def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10

def crop_image(img, width, height):
    h = img.shape[0]
    w = img.shape[1]
    
    i = h - height
    j = int(round((w - width)/2.0))

    return img[i:i+height, j:j+width]

#read the depth map, the code from KITTI offical website
def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    # depth = np.expand_dims(depth,-1) #(x,y) -> (1,x,y)
    return depth


def depth_gt_dilate(depth):
    #dilate the depth map
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
   depth_int = depth.astype(np.int16)#change the depth type to int16
   depth_dilated = cv2.dilate(depth_int, kernel)
   return depth_dilated

if __name__ == "__main__":
   depth_raycast_set = sorted(glob.glob("../raycastdepth/total/*_sync/*.png"))
   depth_pre_prefix_path = "../data/KITTI"
   depth_annotated_prefix_path = "../depth_annotated"
   avg_mae = 0 
   avg_number = 0
   avg_rmse = 0
   avg_absrel = 0
   avg_mask_number=0
   avg_lg10 = 0
   avg_squared_rel = 0
   avg_delta1_125 = 0
   avg_delta2_125 = 0
   avg_delta3_125 = 0
   avg_delta1_101 = 0
   avg_delta2_101 = 0
   avg_delta3_101 = 0

   total_image = len(depth_raycast_set)
   print("total_image: "+str(total_image))
   count = 0

   test_raycast_depth = True
   for depth_path in depth_raycast_set:
       ps = depth_path.split('/')

       sub_dataset = ps[-2]
       depth_index = ps[-1]

       print(sub_dataset)
       depth_pred_path = os.path.join(depth_pre_prefix_path, sub_dataset, "precomputed-depth", depth_index)
       depth_annotated_path = os.path.join(depth_annotated_prefix_path, sub_dataset, "proj_depth/groundtruth/image_02", depth_index)

 #      print(depth_path)
       depth_raycast = depth_read(depth_path)

 #      print(depth_pred_path)
       depth_pred = depth_read(depth_pred_path)

 #      print(depth_annotated_path)
       depth_annotated = depth_read(depth_annotated_path)
       

       depth_raycast = crop_image(depth_raycast, 912, 228)
       depth_pred = crop_image(depth_pred, 912, 228)
       depth_annotated = crop_image(depth_annotated, 912, 228)       


       valid_mask = (depth_raycast > 0.01)*(depth_annotated > 0.01)*(depth_raycast < 50)*(depth_annotated < 50)
       valid_mask1 = (depth_pred > 0.01)*(depth_annotated > 0.01)*(depth_pred < 50)*(depth_annotated < 50)

       mask_nonzeros = np.count_nonzero(valid_mask)


       if test_raycast_depth:
           output_mm = 1e3 * depth_raycast[valid_mask]
           target_mm = 1e3 * depth_annotated[valid_mask]
       else:
           output_mm = 1e3 * depth_pred[valid_mask1]
           target_mm = 1e3 * depth_annotated[valid_mask1]

       abs_diff = np.abs(output_mm-target_mm)
       ##mae
       mae = abs_diff.mean()
       
       ##rmse
       mse = float((abs_diff*abs_diff).mean())
       rmse = math.sqrt(mse)
       
       #lg10
       lg10 = float(np.abs((np.log10(output_mm)-np.log10(target_mm))).mean())

       ##absrel
       absrel = float((abs_diff/target_mm).mean())

       ##squared_rel
       squared_rel = float(((abs_diff/target_mm)**2).mean())

       maxRatio = torch.max(torch.from_numpy(output_mm/target_mm), torch.from_numpy(target_mm/output_mm))
       ##delta1
       delta1_125 = float((maxRatio<1.25).float().mean())
       delta1_101 = float((maxRatio<1.01).float().mean())

       ##delta2
       delta2_125 = float((maxRatio<1.25**2).float().mean())
       delta2_101 = float((maxRatio<1.01**2).float().mean())
       
       ##delta3
       delta3_125 = float((maxRatio<1.25**3).float().mean())
       delta3_101 = float((maxRatio<1.01**3).float().mean())

       avg_mae = avg_mae + mae
       avg_rmse = avg_rmse + rmse
       avg_absrel = avg_absrel + absrel
       #avg_number = avg_number + number_point
       avg_mask_number = avg_mask_number + mask_nonzeros    
       avg_lg10 = avg_lg10 + lg10
       avg_squared_rel = avg_squared_rel + squared_rel
       avg_delta1_125 = avg_delta1_125 + delta1_125
       avg_delta2_125 = avg_delta2_125 + delta2_125
       avg_delta3_125 = avg_delta3_125 + delta3_125

       avg_delta1_101 = avg_delta1_101 + delta1_101
       avg_delta2_101 = avg_delta2_101 + delta2_101
       avg_delta3_101 = avg_delta3_101 + delta3_101

       if count % 50 == 0:
          #print(sub_dataset)
          #print(depth_index)
          print("handle img: "+str(count)+", image: "+str(depth_index)+", mae: "+str(mae)+", rmse: "+str(rmse)+", absrel: "+str(absrel)+", mask_number: "+str(mask_nonzeros))
       count += 1

   avg_mae = avg_mae/total_image
   #avg_number = avg_number/total_image
   avg_rmse = avg_rmse/total_image
   avg_absrel = avg_absrel/total_image
   avg_mask_number = avg_mask_number/total_image
   avg_lg10 = avg_lg10/total_image

   avg_squared_rel = avg_squared_rel/total_image
   avg_delta1_125 = avg_delta1_125/total_image
   avg_delta2_125 = avg_delta2_125/total_image
   avg_delta3_125 = avg_delta3_125/total_image
   avg_delta1_101 = avg_delta1_101/total_image
   avg_delta2_101 = avg_delta2_101/total_image
   avg_delta3_101 = avg_delta3_101/total_image
   
   print("avg mae: "+str(avg_mae)+", avg rmse: "+str(avg_rmse)+", absrel: "+str(avg_absrel)+", mask_number: "+str(avg_mask_number))
   print("avg_lg10: "+str(avg_lg10)+", avg_squared_rel: "+str(avg_squared_rel)+", avg_delta1_125: "+str(avg_delta1_125)+", avg_delta2_125: "+str(avg_delta2_125)+", avg_delta3_125: "+str(avg_delta3_125))
   print("avg_delta1_101: "+str(avg_delta1_101)+", avg_delta2_101: "+str(avg_delta2_101)+", avg_delta3_101: "+str(avg_delta3_101))
       
      
   

