import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import glob
import multiprocessing

# Calibrator class from run_calib.py is integrated here
# ...

# The Calibrator class code should be pasted here
class Calibrator(object):
    def __init__(self, img_dir, shape_inner_corner, size_grid, visualization=True):
        """
        --parameters--
        img_dir: the directory that save images for calibration, str
        shape_inner_corner: the shape of inner corner, Array of int, (h, w)
        size_grid: the real size of a grid in calibrator, float
        visualization: whether visualization, bool
        """
        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None # intrinsic matrix
        self.coff_dis = None # cofficients of distortion

        # create the conner in world space
        w, h = shape_inner_corner
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space
        self.cp_world = cp_int * size_grid

        # images
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(self.img_paths), "No images for calibration found!"

    @staticmethod
    def find_corners(img_path, w, h):
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h),
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        return ret, cp_img

    def calibrate_camera(self):
        w, h = self.shape_inner_corner
        # criteria: only for subpix calibration, which is not used here
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        points_world = [] # the points in world space
        points_pixel = [] # the points in pixel space (relevant to points_world)
        total_images = len(self.img_paths)  # 获取图像总数

        for idx, img_path in enumerate(self.img_paths):  # 使用enumerate获取当前索引
            print(f"Processing image {idx + 1}/{total_images}: {img_path}")  # 打印进度信息和文件名
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"Finding chessboard corners for image {idx + 1}/{total_images}...")
            with multiprocessing.Pool(processes=1) as pool:
                try:
                    ret, cp_img = pool.apply_async(self.find_corners, args=(img_path, w, h)).get(timeout=10)  # 设置10秒超时
                except multiprocessing.TimeoutError:
                    print("Timed out for image: ", img_path)
                    print(f"Error processing image {idx + 1}/{total_images}: multiprocessing.TimeoutError")
            print(f"Done finding chessboard corners for image {idx + 1}/{total_images}.")
            # if ret is True, save
            if ret:
                # cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
                points_world.append(self.cp_world)
                points_pixel.append(cp_img)
                # view the corners
                if self.visualization:
                    # Resize the image to desired size
                    desired_width = 800  # or the width of your screen
                    desired_height = 600  # or the height of your screen
                    scale_x = desired_width / img.shape[1]
                    scale_y = desired_height / img.shape[0]
                    resized_img = resize_image(img, desired_width, desired_height)

                    # Scale the corner points
                    cp_img_resized = cp_img * (scale_x, scale_y)
                    cp_img_resized = np.float32(cp_img_resized)

                    # Draw corners and show the resized image
                    cv2.drawChessboardCorners(resized_img, (w, h), cp_img_resized, ret)
                    cv2.imshow('FoundCorners', resized_img)
                    cv2.waitKey(500)

        # calibrate the camera
        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)
        print ("ret: {}".format(ret))
        print ("intrinsic matrix: \n {}".format(mat_intri))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        print ("distortion cofficients: \n {}".format(coff_dis))
        print ("rotation vectors: \n {}".format(v_rot))
        print ("translation vectors: \n {}".format(v_trans))

        # calculate the error of reproject
        total_error = 0
        for i in range(len(points_world)):
            points_pixel_repro, _ = cv2.projectPoints(points_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis)
            error = cv2.norm(points_pixel[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
            total_error += error
        print("Average error of reproject: {}".format(total_error / len(points_world)))

        self.mat_intri = mat_intri
        self.coff_dis = coff_dis

        # Save the calibration parameters
        self.save_calibration_parameters("calibration_parameters.npz")

        return mat_intri, coff_dis


    def dedistortion(self, save_dir):
        # if not calibrated, calibrate first
        if self.mat_intri is None:
            assert self.coff_dis is None
            self.calibrate_camera()

        w, h = self.shape_inner_corner
        for img_path in self.img_paths:
            _, img_name = os.path.split(img_path)
            img = cv2.imread(img_path)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mat_intri, self.coff_dis, (w,h), 0, (w,h))
            dst = cv2.undistort(img, self.mat_intri, self.coff_dis, None, newcameramtx)
            # clip the image
            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_dir, img_name), dst)
        print("Dedistorted images have been saved to: {}".format(save_dir))

    def save_calibration_parameters(self, filename):
        np.savez_compressed(filename, mat_intri=self.mat_intri, coff_dis=self.coff_dis)

    def load_calibration_parameters(self, filename):
        data = np.load(filename)
        self.mat_intri = data['mat_intri']
        self.coff_dis = data['coff_dis']


    def dedistort_with_saved_parameters(self, img_path, save_path, param_file):
        # Load calibration parameters
        self.load_calibration_parameters(param_file)

        # Check if parameters are loaded
        if self.mat_intri is None or self.coff_dis is None:
            raise ValueError("Calibration parameters are not loaded correctly.")

        # Read the image
        img = cv2.imread(img_path)

        # Get optimal new camera matrix
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mat_intri, self.coff_dis, (w, h), 0, (w, h))

        # Undistort the image
        dst = cv2.undistort(img, self.mat_intri, self.coff_dis, None, newcameramtx)

        # Save the undistorted image
        cv2.imwrite(save_path, dst)

class App:
    def __init__(self, root):
        self.root = root
        root.title("CamCalibTool")

        # Calibration Area
        self.calibration_label = tk.Label(root, text="Calibration")
        self.calibration_label.pack(pady=5)

        self.calibrate_button = tk.Button(root, text="Calibrate Camera", command=self.calibrate_camera)
        self.calibrate_button.pack(pady=5)

        self.save_params_button = tk.Button(root, text="Save Parameters", command=self.save_params)
        self.save_params_button.pack(pady=5)

        # Undistortion Area
        self.undistortion_label = tk.Label(root, text="Undistortion")
        self.undistortion_label.pack(pady=5)

        self.undistort_button = tk.Button(root, text="Undistort Image", command=self.undistort_image)
        self.undistort_button.pack(pady=5)

        self.save_undistorted_button = tk.Button(root, text="Save Undistorted Image", command=self.save_undistorted)
        self.save_undistorted_button.pack(pady=5)

    def calibrate_camera(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Calibration Images")
        # Example: Use the Calibrator class from the repository
        self.calibrator = Calibrator(img_dir=folder_path, shape_inner_corner=(11, 8), size_grid=0.02)
        self.mat_intri, self.coff_dis = self.calibrator.calibrate_camera()

    def save_params(self):
        # Implement parameter saving logic
        self.calibrator.save_calibration_parameters("calibration_parameters.npz")

    def undistort_image(self):
        file_path = filedialog.askopenfilename(title="Select Image to Undistort", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        save_path = filedialog.asksaveasfilename(title="Save Undistorted Image", defaultextension=".png", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        # Example: Use the dedistort_with_saved_parameters method from the Calibrator class
        self.calibrator.dedistort_with_saved_parameters(img_path=file_path, save_path=save_path, param_file="calibration_parameters.npz")

    def save_undistorted(self):
        # Implement saving undistorted image logic
        pass

# Create the main window
root = tk.Tk()
app = App(root)

# Start the app
root.mainloop()
