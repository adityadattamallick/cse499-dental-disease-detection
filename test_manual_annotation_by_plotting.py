import cv2
import os

def yolo_to_xyxy(x_center, y_center, w, h, img_width, img_height):
    """Convert YOLO normalized bbox to pixel (x1, y1, x2, y2)."""
    x1 = int((x_center - w / 2) * img_width)
    y1 = int((y_center - h / 2) * img_height)
    x2 = int((x_center + w / 2) * img_width)
    y2 = int((y_center + h / 2) * img_height)
    return x1, y1, x2, y2

def draw_yolo_bboxes(image_path, label_path, output_path="output.jpg"):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = img.shape

    # Read YOLO labels
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.strip().split())

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        cv2.putText(img, f"class {int(cls)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save output
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")

    # Optionally show image
    cv2.imshow("YOLO BBoxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "image.jpg"
    label_path = "image.txt"
    draw_yolo_bboxes(image_path, label_path)
