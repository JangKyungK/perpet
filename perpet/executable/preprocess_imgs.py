import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os
import glob
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from rembg import remove


# ======================= Resize to 512 =====================================
def resize_image(image_path, output_folderpath, target_size):
    image = cv2.imread(image_path)

    # if image == None:
    #     print("There is no file in raw folder")
    #     return
    height, width, _ = image.shape
    base_path = os.path.basename(image_path)
    name, ext = os.path.splitext(base_path)
    name = f"{name}_resized{ext}"
    output_path = os.path.join(output_folderpath, name)
    # Calculate the scaling factors for width and height
    width_scale = target_size[0] / width
    height_scale = target_size[1] / height
    # Choose the smaller scaling factor to ensure the entire image fits
    scale = min(width_scale, height_scale)
    # Resize the image while maintaining its aspect ratio
    resized_image = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    # Create a blank canvas of the target size (512x512)
    canvas = 255 * np.ones((target_size[1], target_size[0], 3), dtype=np.uint8)
    # Calculate the position to paste the resized image at the center
    x_offset = (canvas.shape[1] - resized_image.shape[1]) // 2
    y_offset = (canvas.shape[0] - resized_image.shape[0]) // 2
    # Paste the resized image onto the canvas
    canvas[
        y_offset : y_offset + resized_image.shape[0],
        x_offset : x_offset + resized_image.shape[1],
    ] = resized_image
    # Save the resized and padded image to the output directory
    cv2.imwrite(output_path, canvas)
    return canvas


# Function to check if any cats are detected
def count_detect_objects(prediction, threshold=0.8):
    # The label for a cat in COCO dataset is 17
    cat_label_id = 17
    dog_label_id = 18
    cat_count = 0
    dog_count = 0
    other_count = 0
    cat_boxes = []
    # Go through all the predictions
    for element in range(len(prediction[0]["labels"])):
        score = prediction[0]["scores"][element]
        if score > threshold:
            label_id = prediction[0]["labels"][element].item()
            if label_id == cat_label_id:
                cat_count += 1
                cat_boxes.append(prediction[0]["boxes"][element].detach().cpu().numpy())
            elif label_id == dog_label_id:
                dog_count += 1
            else:
                other_count += 1
    return cat_count, dog_count, other_count, cat_boxes


def bb_intersection_over_union(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Compute the area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the area of the union
    union_area = boxA_area + boxB_area - intersection_area
    # Compute the IoU
    iou = intersection_area / float(union_area)
    # Return the IoU value
    return iou


# ========================= BG Remove ================================
def bg_remove(input_image_cv2, image_path, output_folderpath):
    cv_image_rgb = cv2.cvtColor(input_image_cv2, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    output = remove(pil_image)
    base_path = os.path.basename(image_path)
    name, ext = os.path.splitext(base_path)
    ext = ".png"
    name = f"{name}_bgRemoved{ext}"
    output_path = os.path.join(output_folderpath, name)
    white_background = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    result = Image.alpha_composite(white_background, output)
    result.save(output_path)


def main(args=None):
    parser = argparse.ArgumentParser(description="Set folder location")
    parser.add_argument(
        "target_name", type=str, help="Base directory to create folders in"
    )

    # Parse command line arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    target_name = args.target_name

    # ------ Debug
    # target_name = "brown_cat"
    # ------

    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))
    target_directory = os.path.join(parent_directory, "customer", target_name)

    raw_folder_path = os.path.join(target_directory, "raw")
    resized_folderpath = os.path.join(raw_folder_path, "resized")

    img_folder_path = os.path.join(target_directory, "img")

    # if image_files != None:
    #     print("Resized photo already exist")
    #     return

    # output_folderpath = os.path.join(target_directory, "img", "")

    # variables
    target_size = (512, 512)

    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Define the image transformation
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    # ============================================================

    img_count = 0
    det_count = 0

    entries = os.listdir(img_folder_path)
    img_folder_list = [
        entry
        for entry in entries
        if os.path.isdir(os.path.join(img_folder_path, entry))
    ]

    for folder in img_folder_list:
        if "xyzctf" in folder:
            continue
        else:
            print("ERROR : img_repeat_folder missing!")

    img_repeat_folder_path = os.path.join(img_folder_path, img_folder_list[0])

    image_files = glob.glob(os.path.join(raw_folder_path, "*.jpg")) + glob.glob(
        os.path.join(raw_folder_path, "*.png")
    )  # Include both .jpg and .png files

    for image_path in image_files:
        img_count += 1
        img = Image.open(image_path)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor[:, :3, :, :]

        with torch.no_grad():
            prediction = model(img_tensor)

        cat_count, dog_count, other_count, cat_boxes = count_detect_objects(prediction)

        if dog_count > 0:
            print(f"Dog detected: {image_path}")
            continue
        elif cat_count > 1:
            print(f"More than one cat detected: {image_path}")
            iou = bb_intersection_over_union(cat_boxes[0], cat_boxes[1])

            print(iou)
            # count as one cat
            if iou > 0.4:
                det_count += 1
                resize_image_cv2 = resize_image(
                    image_path=image_path,
                    output_folderpath=resized_folderpath,
                    target_size=target_size,
                )
                bg_remove(
                    input_image_cv2=input_image_cv2,
                    image_path=image_path,
                    output_folderpath=img_repeat_folder_path,
                )

            # -------Debug - Drawing box
            # for box in cat_boxes:
            #     draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            # img.show()  # or use img.save('path_to_save_detected_image') to save the result
            # ---------

            continue
        elif cat_count == 1 and cat_boxes:
            det_count += 1

            # -------Debug - Drawing box
            # for box in cat_boxes:
            #     draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            # img.show()  # or use img.save('path_to_save_detected_image') to save the result
            # ---------

            input_image_cv2 = resize_image(
                image_path=image_path,
                output_folderpath=resized_folderpath,
                target_size=target_size,
            )
            bg_remove(
                input_image_cv2=input_image_cv2,
                image_path=image_path,
                output_folderpath=img_repeat_folder_path,
            )
        else:
            print(f"Nothing was detected: {image_path}")

    if img_count != det_count:
        try:
            new_repeat_foler_path = os.path.join(
                img_folder_path,
                str(round(1500 / det_count)) + "_xyzctf",
            )
            os.rename(img_repeat_folder_path, new_repeat_foler_path)
        except OSError as error:
            print(f"Error : {error}")


if __name__ == "__main__":
    main()

# print(img_count)
# print(det_count)
