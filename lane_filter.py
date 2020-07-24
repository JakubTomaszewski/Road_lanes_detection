import cv2
import numpy as np


class IncorrectImage(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class LaneFilter:
    """Class containing various functions for image preprocessing"""

    def __init__(self):
        self.white_thresholds = None

    def convert_to_hsv(self, image):
        """Converts given image to HSV color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in HSV color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_to_hsl(self, image):
        """Converts given image to HSL color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in HSL color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    def convert_to_lab(self, image):
        """Converts given image to LAB color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in LAB color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def img_threshold(self, image, threshold, channel_number):
        """Applies a threshold to a specific image color channel

        Parameters
        ----------
        image -- numpy array representing an image

        threshold -- an integer value to threshold the image

        channel_number -- color channel index to be thresholded

        Returns
        -------
        Binary image after thresholding
        """

        if channel_number not in range(image.shape[2]):
            raise IncorrectImage('Insufficient color channels')

        # Setting the channel
        channel = image[:, :, channel_number]

        return cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

    def resize_img(self, image):
        """Reduces the image size

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image of a twice smaller size
        """

        return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    def gaussian_blur(self, image):
        """Applies gaussian blur to an image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image with gaussian blur appied
        """

        # Reducing noise - smoothing
        blurred_img = cv2.GaussianBlur(image, (3, 3), 0)

        return blurred_img

    def get_edges(self, image):
        """Applies few functions to extract edges from an image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------

        """
        kernel = np.ones((5, 5))
        canny = cv2.Canny(image, 100, 200)
        dilate = cv2.dilate(canny, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)
        return erode

    def select_hsl_white_yellow(self, image):
        """Extracts only white and yellow color from an HLS image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        mask -- binary image with extracted only white and yellow color

        masked -- original image with extracted only white and yellow color
        """

        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 255, 255])

        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Yellow mask
        yellow_lower = np.array([10, 0, 100])
        yellow_upper = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return mask, masked

    def select_yellow_hls(self, image, mask_thresholds):
        """Extracts only yellow color from an HLS image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        mask -- binary image with extracted only yellow color
        """

        yellow_lower = np.array(mask_thresholds[0])
        yellow_upper = np.array(mask_thresholds[1])

        # yellow_lower = np.array([0, 150, 10]) # mask_thresholds[0]
        # yellow_upper = np.array([130, 255, 255]) # mask_thresholds[1]

        mask = cv2.inRange(image, yellow_lower, yellow_upper)
        return mask

    def apply_sobel(self, image, channel_number, magnitude_thresh=(50, 210)):
        """Applies sobel operator

        Parameters
        ----------
        image -- numpy array representing an image

        channel_number -- number of the channel to extract

        magnitude_thresh -- threshold for image filtering
            default = (50, 210)

        Returns
        -------
        Binary image with sobel operator applied
        """

        # Setting the channel number
        channel = image[:, :, channel_number]

        # Searching for vertical lines
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0)

        scaled_sobel_x = np.uint8(255 * sobel_x/np.max(sobel_x))

        binary = np.zeros_like(scaled_sobel_x)
        binary[(scaled_sobel_x >= magnitude_thresh[0]) & (scaled_sobel_x <= magnitude_thresh[1])] = 255

        return binary

    def sum_all_binary(self, *args):
        """Applies a bitwise operation and sums all the given images

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        final_image -- binary image with extracted only yellow color
        """

        if len(args) < 2:
            raise IncorrectImage('Insufficient arguments to perform bitwise operation')

        final_image = np.zeros_like(args[0])
        for img in args:
            final_image = cv2.bitwise_or(final_image, img)

        return final_image


class Line:
    """Class representing a line"""

    def __init__(self):

        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients averaged over the last n iterations
        self.best_fit_px = None
        self.best_fit_m = None

        #polynomial coefficients for the most recent fit
        self.current_fit_px = None
        self.current_fit_m = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        # center position of car
        self.lane_to_camera = None

        # Previous Fits
        self.previous_fits_px = []
        self.previous_fits_m = []

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        # meters per pixel in y dimension
        self.ym_per_pix = 30/720

        # y_eval is where we want to evaluate the fits for the line radius calcuation
        # for us it's at the bottom of the image for us, and because we know
        # the size of our video/images we can just hardcode it
        self.y_eval = 720 * self.ym_per_pix

        # camera position is where the camera is located relative to the image
        # we're assuming it's in the middle
        self.camera_position = 640

    def run_line_pipe(self):
        self.calc_best_fit()
        self.calc_radius()

    def add_new_fit(self, new_fit_px, new_fit_m):
        """
        Add a new fit to the Line class
        """

        # If this is our first line, then we will have to take it
        if self.current_fit_px is None and self.previous_fits_px == []:
            self.detected = True
            self.current_fit_px = new_fit_px
            self.current_fit_m = new_fit_m
            self.run_line_pipe()
            return
        else:
            # measure the diff to the old fit
            self.diffs = np.abs(new_fit_px - self.current_fit_px)
            # check the size of the diff
            if self.diff_check():
                print("Found a fit diff that was too big")
                print(self.diffs)
                self.defected = False
                return
            self.detected = True
            self.current_fit_px = new_fit_px
            self.current_fit_m = new_fit_m
            self.run_line_pipe()
            return

    def diff_check(self):
        if self.diffs[0] > 0.001:
            return True
        if self.diffs[1] > 0.25:
            return True
        if self.diffs[2] > 1000.:
            return True
        return False

    def calc_best_fit(self):
        """Calculates the best line fit based on an average"""

        # add the latest fit to the previous fit list
        self.previous_fits_px.append(self.current_fit_px)
        self.previous_fits_m.append(self.current_fit_m)

        # If we currently have 5 fits, throw the oldest out
        if len(self.previous_fits_px) > 5:
            self.previous_fits_px = self.previous_fits_px[1:]
        if len(self.previous_fits_m) > 5:
            self.previous_fits_m = self.previous_fits_m[1:]

        # Just average everything
        self.best_fit_px = np.average(self.previous_fits_px, axis=0)
        self.best_fit_m = np.average(self.previous_fits_m, axis=0)
        return

    def calc_radius(self):
        """left_fit and right_fit are assumed to have already been converted to meters"""

        y_eval = self.y_eval
        fit = self.best_fit_m

        curve_rad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
        self.radius_of_curvature = curve_rad
        return
