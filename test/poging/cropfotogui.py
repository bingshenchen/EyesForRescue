import os
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2

# 加载 YOLO 模型
model = YOLO('best1.4.pt')

# 定义全局变量
current_image_path = None
cropped_img = None
rect_id = None
bbox_coords = None
dragging_corner = None  # 当前拖动的角
is_manual_draw = False  # 是否手动绘制框


def process_image(image_path):
    """
    使用 YOLO 模型检测图像中的目标区域并返回第一个检测框的坐标。
    """
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框
    if len(boxes) == 0:
        return None

    # 返回第一个目标的坐标
    x1, y1, x2, y2 = map(int, boxes[0][:4])
    return (x1, y1, x2, y2)


def display_image(image_path):
    """
    显示图片并绘制初始边框。
    """
    global current_image_path, rect_id, bbox_coords, is_manual_draw

    current_image_path = image_path
    is_manual_draw = False  # 重置手动绘制标志

    # 清除 Canvas 上的所有内容
    canvas.delete("all")

    # 读取图片
    img_cv = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img_pil)

    # 更新 Canvas
    canvas.image = img_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # 获取 YOLO 检测框
    bbox_coords = process_image(image_path)
    if bbox_coords:
        x1, y1, x2, y2 = bbox_coords
        rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    else:
        messagebox.showinfo("Info", f"No objects detected in {image_path}. Please draw a box manually.")
        rect_id = None  # YOLO 没有检测框时，允许用户手动绘制框


def on_press(event):
    """
    鼠标按下事件，用于开始拖动或手动绘制边框。
    """
    global dragging_corner, bbox_coords, is_manual_draw

    x, y = event.x, event.y
    if not bbox_coords:
        # 如果没有检测框，手动绘制新框
        bbox_coords = (x, y, x, y)
        is_manual_draw = True
        update_rectangle()
    else:
        # 检查点击的位置是否靠近边框的某个角
        x1, y1, x2, y2 = bbox_coords
        margin = 10  # 可拖动角的范围
        if abs(x - x1) < margin and abs(y - y1) < margin:
            dragging_corner = "top_left"
        elif abs(x - x2) < margin and abs(y - y1) < margin:
            dragging_corner = "top_right"
        elif abs(x - x1) < margin and abs(y - y2) < margin:
            dragging_corner = "bottom_left"
        elif abs(x - x2) < margin and abs(y - y2) < margin:
            dragging_corner = "bottom_right"
        else:
            dragging_corner = None


def on_drag(event):
    """
    鼠标拖动事件，用于调整边框大小或手动绘制框。
    """
    global rect_id, bbox_coords, dragging_corner, is_manual_draw

    x, y = event.x, event.y
    if is_manual_draw:
        # 更新手动绘制的框的坐标
        x1, y1, _, _ = bbox_coords
        bbox_coords = (x1, y1, x, y)
        update_rectangle()
    elif dragging_corner and bbox_coords:
        # 更新拖动的角的坐标
        x1, y1, x2, y2 = bbox_coords
        if dragging_corner == "top_left":
            bbox_coords = (x, y, x2, y2)
        elif dragging_corner == "top_right":
            bbox_coords = (x1, y, x, y2)
        elif dragging_corner == "bottom_left":
            bbox_coords = (x, y1, x2, y)
        elif dragging_corner == "bottom_right":
            bbox_coords = (x1, y1, x, y)
        update_rectangle()


def update_rectangle():
    """
    更新 Canvas 上的矩形。
    """
    global rect_id, bbox_coords
    if rect_id:
        canvas.coords(rect_id, *bbox_coords)
    else:
        x1, y1, x2, y2 = bbox_coords
        rect_id = canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


def confirm_save(event=None):
    """
    保存裁剪后的图片。
    """
    global current_image_path, bbox_coords

    if current_image_path and bbox_coords:
        x1, y1, x2, y2 = map(int, bbox_coords)
        img_cv = cv2.imread(current_image_path)
        cropped_img = img_cv[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if cropped_img.size > 0:
            # 重命名文件为 cropped_ 开头
            dir_path, original_name = os.path.split(current_image_path)
            new_name = os.path.join(dir_path, f"cropped_{original_name}")
            cv2.imwrite(new_name, cropped_img)
            messagebox.showinfo("Success", f"Image saved as: {new_name}")
        else:
            messagebox.showerror("Error", "Invalid crop area. Skipping save.")
        process_next_image()


def skip_image(event=None):
    """
    跳过当前图片。
    """
    messagebox.showinfo("Info", "Skipping current image.")
    process_next_image()


def process_next_image():
    """
    处理下一张图片。
    """
    global image_paths
    if not image_paths:
        messagebox.showinfo("Info", "All images processed.")
        root.quit()
        return

    next_image = image_paths.pop(0)
    display_image(next_image)


def select_folder():
    """
    让用户选择图片文件夹。
    """
    folder = filedialog.askdirectory(title="Select Image Folder")
    if folder:
        return folder
    else:
        messagebox.showerror("Error", "No folder selected.")
        exit()


if __name__ == "__main__":
    # 选择文件夹
    input_dir = select_folder()

    # 获取文件夹中的所有图片路径
    image_paths = [
        os.path.join(input_dir, file_name)
        for file_name in os.listdir(input_dir)
        if file_name.endswith(".jpg") or file_name.endswith(".png") and not file_name.startswith("cropped_")
    ]

    if not image_paths:
        messagebox.showerror("Error", "No images found in the selected folder.")
        exit()

    # 创建 GUI 窗口
    root = tk.Tk()
    root.title("YOLOv8 Image Cropper")

    # 创建画布用于显示图片
    canvas = tk.Canvas(root, width=2400, height=2400)
    canvas.pack()

    # 绑定鼠标事件

    canvas.bind("<Button-1>", on_press)  # 鼠标按下
    canvas.bind("<B1-Motion>", on_drag)  # 鼠标拖动

    # 绑定按键事件
    root.bind("<Return>", confirm_save)  # 回车键保存
    root.bind("0", skip_image)  # 按键 0 跳过

    # 显示第一张图片
    process_next_image()

    # 运行 GUI
    root.mainloop()
