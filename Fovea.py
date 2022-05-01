# Main file for the file iteration
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
myList = [] ## area list
classList = [] ##class Id List

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


def save_image(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
               scores_thresh=0.1, save_dir=None, mode=0):
    """
        image: image array
        image_name: image name
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        save_dir: (optional) the path to store image
        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
    mode_list = [0, 1, 2, 3]
    assert mode in mode_list, "mode's value should in mode_list %s" % str(mode_list)

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    useful_mask_indices = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        useful_mask_indices.append(i)

    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))
        return

    colors = random_colors(len(useful_mask_indices))

    if mode != 3:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)

    if mode != 1:
        for index, value in enumerate(useful_mask_indices):
            masked_image = apply_mask(masked_image, masks[:, :, value], colors[index])

    masked_image = Image.fromarray(masked_image)

    if mode == 3:
        masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
        return

    draw = ImageDraw.Draw(masked_image)
    colors = np.array(colors).astype(int) * 255

    myList = []
    countClassIds = 0
    
    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        print('class_id value is {}'.format(class_id))
        if class_id == 1:
          countClassIds += 1
        print('counter for the class ID {}'.format(countClassIds))
        
        
        score = scores[value]
        label = class_names[class_id]

        y1, x1, y2, x2 = boxes[value]
        
#         myList = []
        
        ## area of the rectangle
        yVal = y2 - y1
        xVal = x2 - x1
        area = xVal * yVal
        print('area is {}'.format(area))
        myList.append(area)
        
        if mode != 2:
            color = tuple(colors[index])
            draw.rectangle((x1, y1, x2, y2), outline=color)

        # Label
#         font = ImageFont.load('/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf')
        font = ImageFont.truetype('OpenSans-Bold.ttf', 15)
        draw.text((x1, y1), "%s %f" % (label, score), (255, 255, 255), font)

    print(r['class_ids'], r['scores'])
    print(myList)
#     print('value of r is {}'.format(r))
    print('image_name is {}'.format(image_name))

    image_name = os.path.basename(image_name)
    print('image name is {}'.format(image_name))

    f_name, f_ext = os.path.splitext(image_name)
    #f_lat_val,f_long_val,f_count = f_name.split('-')

    #f_lat_val = f_lat_val.strip() ##removing the white Space
    #f_long_val = f_long_val.strip()

#     new_name = '{}-{}-{}.jpg'.format(f_lat_val,f_long_val,count)
#     print([area for area in myList if ])
#       print([i for i in range(countClassIds) ])
      
    print("avi96 {}".format(myList[:countClassIds]))
#     myList.pop(countClassIds - 1)
    
    new_name = '{}-{}.jpg'.format(myList, r['scores'])
#     masked_image.save(os.path.join(save_dir, '%s.jpg' % (image_name)))
    print("New Name file is {}".format(new_name))
    print('save_dir is {}'.format(save_dir))
    masked_image.save(os.path.join(save_dir, '%s' % (new_name)))
    print('file Saved {}'.format(new_name))
#     os.rename(image_name, new_name)



if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys
    import custom
    import utils
    import model as modellib
    #import visualize

    # We use a K80 GPU with 24GB memory, which can fit 3 images.
    batch_size = 3

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
    VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
#     COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_damage_0010.h5")
#     if not os.path.exists(COCO_MODEL_PATH):
#         utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(custom.CustomConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = batch_size

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights("logs/mask_rcnn_damage_0160.h5", by_name=True)
    class_names = [
        'BG', 'damage'
    ]

#     capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'trailer1.mp4'))
    try:
        if not os.path.exists(VIDEO_SAVE_DIR):
            os.makedirs(VIDEO_SAVE_DIR)
    except OSError:
        print ('Error: Creating directory of data')

    # points to be done before final coding
    """
    path_for_image_dir
    list for the image array
    resolve for naming convention for location basis
    passing image in model
    """

    # path for the data files
    data_path = '/content/Mask_RCNN/S3_Images/images/'
    onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    # empty list for the training data
    frames = []
    frame_count = 0
    batch_count = 1

    # enumerate the iteration with number of files
    for j, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[j]
#         print("image Path {}".format(image_path))
#         print("Only Files {}".format(onlyfiles[j]))
#         print('j is {}'.format(j))
#         print('files is {}'.format(files))
        try:
            images = cv2.imread(image_path).astype(np.uint8)
#             print("images {}".format(images))
            frames.append(np.asarray(images, dtype=np.uint8))
  #         frames.append(images)
            frame_count += 1
            print('frame_count :{0}'.format(frame_count))
            if len(frames) == batch_size:
                results = model.detect(frames, verbose=0)
                print('Predicted')
                for i, item in enumerate(zip(frames, results)):
#                     print('i is {}'.format(i))
#                     print('item is {}'.format(item))
                    frame = item[0]
                    r = item[1]
                    frame = display_instances(
                      frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                    )
                    name = '{}'.format(files)
                    name = os.path.join(VIDEO_SAVE_DIR, name)
#                     name = '{0}.jpg'.format(frame_count + i - batch_size)
#                     name = os.path.join(VIDEO_SAVE_DIR, name)
#                   cv2.imwrite(name, frame)
#                     print(name)
                    print('writing to file:{0}'.format(name))
#                     print(name)
                    save_image(images, name, r['rois'], r['masks'], r['class_ids'],
                             r['scores'], class_names, save_dir=VIDEO_SAVE_DIR, mode=0)
                frames = []
                print('clear')
              # clear the frames here

        except(AttributeError) as e:
            print('Bad Image {}'.format(image_path))

    print("Success, check the folder")


"""
    ## Code for the video section
    frames = []
    frame_count = 0
    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)
        print('frame_count :{0}'.format(frame_count))
        if len(frames) == batch_size:
            results = model.detect(frames, verbose=0)
            print('Predicted')
            for i, item in enumerate(zip(frames, results)):
                frame = item[0]
                r = item[1]
                frame = display_instances(
                    frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
                )
#                 name = '{0}.jpg'.format(frame_count + i - batch_size)
#                 name = os.path.join(VIDEO_SAVE_DIR, name)
#                 cv2.imwrite(name, frame)
#                 print('writing to file:{0}'.format(name))
                ## add visualise files
#                 visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
                save_image(image, name, r['rois'], r['masks'], r['class_ids'],
                                     r['scores'],class_names, save_dir=VIDEO_SAVE_DIR, mode=0)
#                 print(r['class_ids'], r['scores'])

            # Clear the frames array to start the next batch
            frames = []

    capture.release()
"""