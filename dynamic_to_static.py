import runOpticalFlow
import runSegmentation
import runBlurDetection
import determineHomography as runHomography

import arguments
from glob import glob
import os.path as osp
import os
from functools import lru_cache
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time


@lru_cache(maxsize=250)
def read_image(img_path):
    return cv2.imread(img_path)

def generateImages(dirname, startIdx, endIdx):
    paths = Path(dirname).glob('*.png')
    paths = [str(path) for path in paths]
    paths.sort()
    return paths[startIdx:endIdx+1]

# Process the images
def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    files = generateImages(dirname, start_idx, end_idx)
    assert len(files) > 1, f"Not enough images to generate pairs, only {len(files)} images"
    lastImage = files[0]
    for image in files[1:]:
      img_pairs.append((lastImage, image))
      lastImage = image

    return img_pairs


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.legacy.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.legacy.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.legacy.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.legacy.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.legacy.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.legacy.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.legacy.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.legacy.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
 
  return tracker

class HumanDetracker:
  def __init__(self, segResult):
     self.maskImage = np.ones((480, 640), dtype=np.bool_)
     for result in segResult:
        if result.class_name != 'person':
            continue
        x, y, w, h = result.bbox
        self.maskImage[y:h, x:w] = result.mask[y:h, x:w] == 0
      
  def __call__(self, image, geo_bbox):
    image = cv2.bitwise_and(image, image, mask=self.maskImage.astype(np.uint8))
    for bbox in geo_bbox:
      x1, y1, x2, y2 = bbox
      image[y1:y2, x1:x2] = 0      
    return image

class MultiTracker:
  def __init__(self, segResult, image):
    self.multiTracker = cv2.legacy.MultiTracker_create()
    self.masks = []
    for result in segResult:
      if result.class_name != 'person':
          continue
      x, y, w, h = result.bbox
      mask = result.mask[y:h, x:w].astype(np.uint8)*255
      self.masks.append(mask)
      self.multiTracker.add(createTrackerByName('KCF'), image, result.bbox)

  def __call__(self, image):
    maskImage = np.zeros((480, 640), dtype=np.uint8)
    self.multiTracker.update(image)
    for mask, tracker in zip(self.masks, self.multiTracker.getObjects()):
      x, y, w, h = tracker
      x, y, w, h = int(x), int(y), int(w), int(h)
      assert maskImage[y:y+h, x:x+w].shape == mask.shape, f"maskImage shape: {maskImage[y:y+h, x:x+w].shape}, mask shape: {mask.shape}"
      maskImage[y:y+h, x:x+w] = cv2.bitwise_and(maskImage[y:y+h, x:x+w], mask)
    return maskImage
   
class IndividualTracker:
  def __init__(self, segResult, image):
    self.templates = []
    self.masks= []
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for result in segResult:
      if result.class_name != 'person':
          continue
      x, y, w, h = result.bbox
      templateImage = imageGray[y:h, x:w]
      mask = result.mask[y:h, x:w].astype(np.uint8)*255
      self.templates.append(templateImage)
      self.masks.append(mask)

  def __call__(self, image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    maskImage = np.zeros((480, 640), dtype=np.uint8)
    for mask, templateImage in zip(self.masks, self.templates):
      res = cv2.matchTemplate(imageGray, templateImage, cv2.TM_CCOEFF_NORMED)
      _, _, _, max_loc = cv2.minMaxLoc(res)
      x, y, w, h = max_loc[0], max_loc[1], templateImage.shape[1], templateImage.shape[0]
      assert maskImage[y:y+h, x:x+w].shape == mask.shape, f"maskImage shape: {maskImage[y:y+h, x:x+w].shape}, mask shape: {mask.shape}"
      maskImage[y:y+h, x:x+w] = cv2.bitwise_and(maskImage[y:y+h, x:x+w], mask)
    return maskImage

class Pipeline:
  def __init__(self, image, args, modelSeg):
    self.modelSeg = modelSeg
    self.index = 0
    self.reset(image, args)
   
  def __call__(self, image, args):
    self.index += 1
    if self.index % args.template_value == 0:
      return self.reset(image, args)
    if self.index == 2:
      self.geoBbox = runHomography.geometry_evaluation(self.lastImage, image, args.eps_value)
    imageChanged = self.tracker(image, self.geoBbox)
    self._lastImage = image
    return imageChanged

  def reset(self, image, args):
    with torch.no_grad():
      _, self.segResult = runSegmentation.evalimage(args, self.modelSeg, image)
    self.tracker = HumanDetracker(self.segResult)
    if self.index > 0:
      self.geoBbox = runHomography.geometry_evaluation(self.lastImage, image, args.eps_value)
    self.index += 1
    self._lastImage = image
    self._lastDehumanImage = self.tracker(image, [])
    return self._lastDehumanImage
  
  @property
  def lastDehumanImage(self):
    return self._lastDehumanImage

  @property
  def lastImage(self):
    return self._lastImage


def main():
    run_detection = False

    args = arguments.argparser()
    dir_name = args.root_dir/args.seq_dir

    output_dir_name = str(args.seq_dir).split('/')[-2] + "_" + str(args.eps_value)

    if torch.cuda.is_available():
        args.cuda = True

    if args.cuda:
        # cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    args.output_dir.mkdir(exist_ok=True, parents=True)

    rgb_output = args.output_dir/output_dir_name/'rgb'
    gray_output = args.output_dir/output_dir_name/'gray'

    rgb_output.mkdir(exist_ok=True, parents=True)
    gray_output.mkdir(exist_ok=True, parents=True)

    # Build the models, if required
    if args.run_flowformer:
        modelOptical = runOpticalFlow.build_model()
        print("Finished loading the optical flow model.")
    
    if args.run_yolact or True:
        modelSeg = runSegmentation.load_model(args)
        print("Finised loading the segmentation model.")

    if args.run_blur_detection:
        run_detection = True

    if args.run_full_system:
        print(f"running full system...")
        count_images = len(sorted(glob(osp.join(dir_name, '*.png'))))
        args.end_idx = count_images

    img_pairs = generate_pairs(dir_name, args.start_idx, args.end_idx)
    
    optical_flow_time = []
    segmentation_time = []
    blur_detection_time = []
    tracker_time = []
    template_time = []
    homography_time = []

    def save_image(in_path, image):
      if not args.save_images:
        return
      filename = Path(in_path).name
      cv2.imwrite(str(rgb_output/filename), image)
      cv2.imwrite(str(gray_output/filename), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if args.save_video:
        vw_rgb = cv2.VideoWriter(str(args.output_dir/output_dir_name/'output.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 480))
        vw_gray = cv2.VideoWriter(str(args.output_dir/output_dir_name/'output_gray.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 480), False)

    def save_video(image):
      if not args.save_video:
        return
      vw_rgb.write(image)
      vw_gray.write(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    imageList = generateImages(dir_name, args.start_idx, args.end_idx)
    
    pipeline = Pipeline(read_image(imageList[0]), args, modelSeg)
    save_image(imageList[0], pipeline.lastDehumanImage)
    save_video(pipeline.lastDehumanImage)

    for imagePath in tqdm(imageList[1:], total=len(imageList)-1):
        currentImage = read_image(imagePath)
        imageChanged = pipeline(currentImage, args)
        save_image(imagePath, imageChanged)
        save_video(imageChanged)


    # for i, image in tqdm(enumerate(img_pairs), total=len(img_pairs)):
    #     image1 = read_image(image[0])
    #     image2 = read_image(image[1])

    #     if args.run_flowformer:
    #         with torch.no_grad():
    #             start_time = time.time()
    #             optical_flow = runOpticalFlow.visualize_flow(image1, image2, modelOptical, args.keep_size)
    #             optical_flow_time.append(time.time() - start_time)

    #     # Run through segmentation
    #     if args.run_yolact:
    #         start_time = time.time()
    #         with torch.no_grad():
    #             if i == 0:
    #               _, segResult1 = runSegmentation.evalimage(args, modelSeg, image1)
    #             _, segResult2 = runSegmentation.evalimage(args, modelSeg, image2)

    #         segmentation_time.append(time.time() - start_time)
        
    #     if args.run_homography:
    #         start_time = time.time()
    #         geo_bbox = runHomography.geometry_evaluation(image1, image2, args.eps_value)
    #         homography_time.append(time.time() - start_time)

    #     if run_detection:
    #         start_time = time.time()
    #         blur1 = runBlurDetection.detect_blur(image1)
    #         blur2 = runBlurDetection.detect_blur(image2)
    #         # print(f"Blur1: {blur1}, Blur2: {blur2}")
    #         blur_detection_time.append(time.time() - start_time)
        
    #     if args.run_tracker:
    #       start_time = time.time()
    #       tracker = IndividualTracker(segResult1, image1)
    #       mask = tracker(image2)
        

    #     # Use Seg bounding box and geometric bbox to determine the final bbox
    #     if i == 0:
    #       # in image1, use segResult1.bbox and geo_bbox to make the corresponding pixel in image1 to be 0
    #       humanDetracker = HumanDetracker(segResult1)
    #       image_changed = humanDetracker(image1, [])
    #       # Save the image
    #       output_name = Path(image[0]).name
    #       cv2.imwrite(str(rgb_output/output_name), image_changed)
    #       cv2.imwrite(str(gray_output/output_name), cv2.cvtColor(image_changed, cv2.COLOR_BGR2GRAY))

    #     output_name = Path(image[1]).name
    #     humanDetracker = HumanDetracker(segResult2)
    #     image_modif = humanDetracker(image2, geo_bbox)
    #     cv2.imwrite(str(rgb_output/output_name), image_modif)
    #     cv2.imwrite(str(gray_output/output_name), cv2.cvtColor(image_modif, cv2.COLOR_BGR2GRAY))

    print("Number of times run: ", len(optical_flow_time))

    print(f"Optical flow time: {np.mean(optical_flow_time)}")
    print(f"Segmentation time: {np.mean(segmentation_time)}")
    print(f"Blur detection time: {np.mean(blur_detection_time)}")
    print(f"Template time: {np.mean(template_time)}")
    #print(f"Tracker time: {np.mean(tracker_time)}")
    print(f"Homography time: {np.mean(homography_time)}")
    
if __name__ == '__main__':
    main()

    #       template_time.append(time.time() - start_time)

    #         # start_time = time.time()
    #         # tracker = MultiTracker(segResult1, image1)
    #         # mask = tracker(image2)
    #         # tracker_time.append(time.time() - start_time)