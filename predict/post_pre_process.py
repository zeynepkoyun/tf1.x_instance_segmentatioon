
import uuid,os
import collections
import matplotlib
#for windows
#matplotlib.use('TkAgg')
#fro ubuntu
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont



class PostProcess():


    STANDARD_COLORS1 = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    STANDARD_COLORS = [
         'Cyan', 'Pink', 'Purple','Brown',
        'Red','Yellow', 'Green','Blue','orange',
    ]

    @staticmethod
    def draw_bounding_box_on_image(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=(),
                                   use_normalized_coordinates=True):

        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                  text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin

    @staticmethod
    def draw_bounding_box_on_image_array(image,
                                         ymin,
                                         xmin,
                                         ymax,
                                         xmax,
                                         color='red',
                                         thickness=4,
                                         display_str_list=(),
                                         use_normalized_coordinates=True):

        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        PostProcess.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                   thickness, display_str_list,
                                   use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

    @staticmethod
    def draw_mask_on_image_array(img_name, image, mask, box=None, color='red', alpha=0.4):
        """Draws mask on an image.

        Args:
          image: uint8 numpy array with shape (img_height, img_height, 3)
          mask: a uint8 numpy array of shape (img_height, img_height) with
            values between either 0 or 1.
          color: color to draw the keypoints with. Default is red.
          alpha: transparency value between 0 and 1. (default: 0.4)

        Raises:
          ValueError: On incorrect data type for image or masks.
        """
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.uint8:
            raise ValueError('`mask` not of type np.uint8')
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                             'dimensions %s' % (image.shape[:2], mask.shape))
        rgb = ImageColor.getrgb(color)

        pil_image = Image.fromarray(image)
        pil_image_first = Image.fromarray(image)

        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')


        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask))  # .convert('L')  # L

        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)

        np.copyto(image, np.array(pil_image.convert('RGB')))
        x1 = int(box[0] * image.shape[0])
        y1 = int(box[1] * image.shape[1])
        x2 = int(box[2] * image.shape[0])
        y2 = int(box[3] * image.shape[1])
        mask_inv=cv2.bitwise_not(mask)
        bg = np.zeros((image.shape),dtype=image.dtype)
        bg.fill(255)
        bgBlack=np.zeros((image.shape),dtype=image.dtype)
        bgBlack.fill(0)

        new2=cv2.bitwise_and(bg,bg,mask=mask)
        new2=cv2.bitwise_not(new2)

        newimage = cv2.bitwise_and(image, image, mask=mask)

        new3=cv2.add(newimage,new2)

        new4=cv2.cvtColor(new3,cv2.COLOR_BGR2RGB)
        new4 = new4[x1:x2, y1:y2]


        # cv2.imwrite("new{}.jpg".format(img_name),new4)

        bg = Image.new('RGB', (y2, x2), (255, 255, 255))

        bg.paste(pil_image_first)
        area = (y1, x1, y2, x2)  # (x1,y1,y2,x2)
        cropped_img = bg.crop(area)
        id_val = str(uuid.uuid4())

        # img_new_path = "output{}.jpg".format(id_val)
        # cropped_img.save(img_new_path)

        #
        # ymin, xmin, ymax, xmax = box


        # x1, y1, x2, y2 = box# -10, -20, 1000, 500  # cropping coordinates
        # print(box)
        # print(x1,y1,x2,y2)

        # bg = Image.new('RGB', (y2,x2), (255, 255, 255))
        # pil_image1 = Image.composite(pil_solid_color, bg, pil_mask) #pil_image pil image #sadece maske icin pil_solid_color
        # area = (y1,x1,y2,x2) #(x1,y1,y2,x2)
        # cropped_img = pil_image1.crop(area)
        # cropped_img.show()
        # # bg.paste(pil_solid_color, (-x1, -y1))
        # cropped_img.save("output.jpg")

        # bg = Image.new('RGB', (y2, x2), (255, 255, 255))
        #
        # bg.paste(pil_image_first)
        # area = (y1, x1, y2, x2)  # (x1,y1,y2,x2)
        # cropped_img = bg.crop(area)
        # # cropped_img.show()
        # cropped_img.save("output.jpg")

        # import numpy
        # from PIL import  ImageDraw
        #
        # # read image as RGB and add alpha (transparency)
        # im = Image.fromarray(image).convert("RGBA")
        #
        # # convert to numpy (for convenience)
        # imArray = numpy.asarray(im)
        #
        # # create mask
        # polygon = mask
        # maskIm = pil_mask
        # mask = numpy.array(maskIm)
        #
        # # assemble new image (uint8: 0-255)
        # newImArray = numpy.empty(imArray.shape, dtype='uint8')
        #
        # # colors (three first columns, RGB)
        # newImArray[:, :, :3] = imArray[:, :, :3]
        #
        # # transparency (4th column)
        # newImArray[:, :, 3] = mask * 255
        #
        # # # back to Image from numpy
        # newIm = Image.fromarray(newImArray, "RGBA")
        # newIm.save("outz.png","PNG")
        #
        # im=Image.open("outz.png")
        # bg = Image.new("RGB", im.size, (255, 255, 255))
        # bg.paste(im, im)
        # area = (y1, x1, y2, x2)  # (x1,y1,y2,x2)
        # cropped_img = bg.crop(area)
        # cropped_img.save("outz.jpg")

        # bg.save("outz.jpg")

    @staticmethod
    def visualize_boxes_and_labels_on_image_array(
            detection_results_image_path,
            image_name,
            image,
            boxes,
            classes,
            scores,
            category_index,
            im_width,
            im_height,
            instance_masks=None,
            instance_boundaries=None,
            keypoints=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.0,
            agnostic_mode=False,  # False
            line_thickness=4,
            groundtruth_box_visualization_color='white',
            skip_scores=False,
            skip_labels=False):

        try:
            box_to_display_str_map = collections.defaultdict(list)
            box_to_display_str_class = collections.defaultdict(list)
            box_to_display_str_score = collections.defaultdict(list)
            box_to_color_map = collections.defaultdict(str)
            box_to_instance_masks_map = {}
            box_to_instance_boundaries_map = {}
            box_to_keypoints_map = collections.defaultdict(list)
            if not max_boxes_to_draw:
                max_boxes_to_draw = boxes.shape[0]
            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                if scores is None or scores[i] > min_score_thresh:
                    box = tuple(boxes[i].tolist())
                    if instance_masks is not None:
                        box_to_instance_masks_map[box] = instance_masks[i]
                    if instance_boundaries is not None:
                        box_to_instance_boundaries_map[box] = instance_boundaries[i]
                    if keypoints is not None:
                        box_to_keypoints_map[box].extend(keypoints[i])
                    if scores is None:
                        box_to_color_map[box] = groundtruth_box_visualization_color
                    else:
                        display_str = ''
                        if not skip_labels:
                            if not agnostic_mode:
                                if classes[i] in category_index.keys():
                                    class_name = category_index[classes[i]]['name']
                                else:
                                    # print("classes[i]", classes[i])
                                    class_name = 'N/A'
                                display_str = str(class_name)
                        if not skip_scores:
                            if not display_str:
                                display_str = '{}%'.format(scores[i])
                            else:
                                display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                        box_to_display_str_map[box].append(display_str)
                        box_to_display_str_class[box].append(class_name)
                        box_to_display_str_score[box].append(scores[i])
                        if agnostic_mode:
                            box_to_color_map[box] = 'DarkOrange'
                        else:
                            box_to_color_map[box] = PostProcess.STANDARD_COLORS[
                                classes[i] % len(PostProcess.STANDARD_COLORS)]

            # Draw all boxes onto image.
            i = -1
            # print(box_to_color_map)
            for box, color in box_to_color_map.items():
                ymin, xmin, ymax, xmax = box

                left, right, top, bottom = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                print("b", box, box_to_display_str_class[box], box_to_display_str_score[box])
                print(color)
                if instance_masks is not None:
                    # print("instance_masks girdi")
                    i += 1
                    segmentation = PostProcess.draw_mask_on_image_array(
                        i,
                        image,
                        box_to_instance_masks_map[box],
                        box,
                        color=color
                    )

                    # print("--- ", os.path.join(detection_results_image_path, (image_name + ".txt")))
                    with open(os.path.join(detection_results_image_path, (image_name + ".txt")), "a") as new_f:
                        new_f.write("%s %s %s %s %s %s\n" % (
                        box_to_display_str_class[box][0], box_to_display_str_score[box][0], left, top, right, bottom))

                #bbox kapatmak
                PostProcess.draw_bounding_box_on_image_array(
                    image,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color=color,
                    thickness=line_thickness,
                    display_str_list=box_to_display_str_map[box],
                    use_normalized_coordinates=use_normalized_coordinates)

            return image

        except Exception as ex:
            print("--- HATA (post_pre_process/visualize_boxes_and_labels_on_image_array)---",ex)

class PreProcess():

    @staticmethod
    def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                         image_width):
        # TODO(rathodv): Make this a public function.
        def reframe_box_masks_to_image_masks_default():
            """The default function when there are more than 0 box masks."""

            def transform_boxes_relative_to_boxes(boxes, reference_boxes):
                boxes = tf.reshape(boxes, [-1, 2, 2])
                min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
                max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
                transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
                return tf.reshape(transformed_boxes, [-1, 4])

            box_masks_expanded = tf.expand_dims(box_masks, axis=3)
            num_boxes = tf.shape(box_masks_expanded)[0]
            unit_boxes = tf.concat(
                [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
            reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
            return tf.image.crop_and_resize(
                image=box_masks_expanded,
                boxes=reverse_boxes,
                box_ind=tf.range(num_boxes),
                crop_size=[image_height, image_width],
                extrapolation_value=0.0)

        image_masks = tf.cond(
            tf.shape(box_masks)[0] > 0,
            reframe_box_masks_to_image_masks_default,
            lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
        return tf.squeeze(image_masks, axis=3)

