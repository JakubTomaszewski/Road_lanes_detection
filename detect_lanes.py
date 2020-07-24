import numpy as np
import cv2
import os
from warper import ImageWarper
from lane_filter import LaneFilter, Line
from trackbar import initialize_trackbar, initialize_threshold_trackbar, val_trackbar, draw_points, get_thresh_trackbar_vals
from script import (combine_radius,
                    calc_line_fits_from_prev,
                    calc_line_fits,
                    get_center_dist,
                    create_final_image,
                    add_image_text)


def get_img_names(path=None):
    """Lists all .png and .jpg files from a directory

    Parameters
    ----------
    path -- path do a directory
        default=None

    Returns
    -------
    Lists of all .png and .jpg images
    """

    names = []
    try:
        for file in os.listdir(path):
            if file.endswith('.png') or file.endswith('jpg'):
                names.append(file)
    except OSError:
        print('Incorrect path')
    return names


def main():
    WARP_TRACKBAR_NAME = 'Warp Trackbar'

    cap = cv2.VideoCapture('test_videos/project_video.mp4')

    img_filter = LaneFilter()
    initialize_trackbar([760, 450, 1150, 650], WARP_TRACKBAR_NAME, 1280)
    initialize_threshold_trackbar(((0, 150, 10), (130, 255, 255)), 'Threshold Trackbar', 255)

    ym_per_pix = 30/720
    xm_per_pix = 3.7/550
    leftLine = Line()
    rightLine = Line()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        height = frame.shape[0]
        width = frame.shape[1]

        final_image = frame.copy()

        src_pts = val_trackbar(frame, WARP_TRACKBAR_NAME)
        dst_pts = np.array([(0, height), (0, 0), (width, 0), (width, height)], dtype=np.float32)

        points = draw_points(frame.copy(), src_pts)

        img_warper = ImageWarper(src_pts, dst_pts)

        warped = img_warper.warp_matrix(frame)

        preprocessed_img = img_filter.gaussian_blur(warped)

        _, thresh_rgb_r = img_filter.img_threshold(preprocessed_img, 190, 0)

        hls = img_filter.convert_to_hsl(preprocessed_img)

        # sobel = img_filterapply_sobel(hls, 1)

        mask_thresholds = get_thresh_trackbar_vals('Threshold Trackbar')

        yellow_mask = img_filter.select_yellow_hls(preprocessed_img, mask_thresholds)

        masked_image = img_filter.sum_all_binary(thresh_rgb_r, yellow_mask)

        resized_images = list(map(img_filter.resize_img, (thresh_rgb_r, yellow_mask, masked_image)))

        vert = np.concatenate(resized_images)

        try:
            # If we found lines previously, run the simplified line fitter
            if leftLine.detected is True and rightLine.detected is True:
                left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits_from_prev(masked_image,
                                                                                                 masked_image,
                                                                                                 leftLine, rightLine,
                                                                                                 ym_per_pix,
                                                                                                 xm_per_pix)
            else:
                # Run the warped, binary image from the pipeline through the complex fitter
                left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(masked_image,
                                                                                       ym_per_pix,
                                                                                       xm_per_pix)
        except TypeError:
            pass

        # Add these fits to the line classes
        leftLine.add_new_fit(left_fit, left_fit_m)
        rightLine.add_new_fit(right_fit, left_fit_m)

        # get radius and center distance
        curve_rad = combine_radius(leftLine, rightLine)
        head, center_distance_m, center_distance_px = get_center_dist(leftLine, rightLine, xm_per_pix)

        # create the final image
        result = create_final_image(frame, masked_image, leftLine, rightLine, img_warper)
        
        # add the text to the image
        result = add_image_text(result, curve_rad, head, center_distance_m, center_distance_px)

        warp_pipeline = np.concatenate(list(map(img_filter.resize_img, (warped, points))))
        cv2.imshow('perspective', warp_pipeline)

        cv2.imshow('vert', vert)

        cv2.imshow('lanes_image', img_filter.resize_img(out_img))
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
