import numpy as np
import cv2
import os
from warper import ImageWarper
from lane_filter import LaneFilter, Line
from trackbar import initialize_trackbar, val_trackbar, draw_points


def get_img_names(path=None):
    """

    :param path:
    :return:
    """

    names = []
    try:
        for file in os.listdir(path):
            if file.endswith('.png') or file.endswith('jpg'):
                names.append(file)
    except OSError:
        print('Incorrect path')
    return names


def sum_all_binary(*args):
    if len(args) < 2:
        print('Insufficient arguments to perform bitwise operation')
        return -1
    #     raise Error later

    final_image = np.zeros_like(args[0])
    for img in args:
        final_image = cv2.bitwise_or(final_image, img)

    return final_image


def hist_calculation(image):
    hist = np.sum(image[image.shape[0]//2:, :], axis=0)

    midpoint = image.shape[0] // 2
    left_x_base = np.argmax(hist[:midpoint])
    right_x_base = np.argmax(hist[midpoint:]) + midpoint

    return midpoint, left_x_base, right_x_base


def calc_line_fits(img, ym_per_pix=30/720, xm_per_pix=3.7/550):

    # # meters per pixel in y dimension
    # ym_per_pix = 30/720
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # ym_per_pix = 3.0/100 # meters per pixel in y dimension, lane line is 10 ft = 3 meters
    # xm_per_pix = 3.7/550  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters

    # Settings
    # Choose the number of sliding windows (for these images, the height is 450)
    nwindows = 10  # or 10
    # Set the width of the windows +/- margin
    margin = 130
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img


def calc_line_fits_from_prev(img, binary_warped, leftLine, rightLine, ym_per_pix=30/720, xm_per_pix=3.7/550):

    left_fit = leftLine.best_fit_px
    right_fit = rightLine.best_fit_px

    ### Settings
    margin = 100  # Width on either side of the fitted line to search

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each in meters
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return left_fit, right_fit, left_fit_m, right_fit_m, result


def calc_radius(y_eval, left_fit, right_fit, ym_per_pix):
    """
    left_fit and right_fit are assumed to have already been converted to meters
    """
    y_eval = y_eval * ym_per_pix
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad


def get_center_dist(leftLine, rightLine, xm_per_pix):

    # grab the x and y fits at px 700 (slightly above the bottom of the image)
    y = 700
    image_center = 640
    image_center_meters = image_center * xm_per_pix

    leftPos = leftLine.best_fit_px[0]*(y**2) + leftLine.best_fit_px[1]*y + leftLine.best_fit_px[2]
    rightPos = rightLine.best_fit_px[0]*(y**2) + rightLine.best_fit_px[1]*y + rightLine.best_fit_px[2]
    lane_middle = int((rightPos - leftPos) / 2) + leftPos
    lane_middle_meters = lane_middle * xm_per_pix

    mag_in_pixels = lane_middle - image_center

    mag_in_meters = lane_middle_meters - image_center_meters

    if mag_in_meters > 0:
        head = "Right"
    else:
        head = "Left"

    return head, mag_in_meters, mag_in_pixels


def combine_radii(leftLine, rightLine):

    left = leftLine.radius_of_curvature
    right = rightLine.radius_of_curvature

    return np.average([left, right])


def create_final_image(img, binary_warped, leftLine, rightLine, warper, show_images=False):

    left_fit = leftLine.best_fit_px
    right_fit = rightLine.best_fit_px

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper.inverse_warp_matrix(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)

    return result


def add_image_text(img, radius, head, center_m, center_px):

    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_DUPLEX

    text = 'Radius of curvature: ' + '{:04.0f}'.format(radius) + 'm'
    cv2.putText(img, text, (50,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    text = '{:03.2f} m {} of center'.format(abs(center_m), head)
    cv2.putText(img, text, (50, 175), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    text = '{:03.2f} pixels {} of center'.format(abs(center_px), head)
    cv2.putText(img, text, (50, 220), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def main():
    cap = cv2.VideoCapture('test_videos/project_video.mp4')

    img_filter = LaneFilter()
    initialize_trackbar([760, 450, 1300, 650])

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

        src_pts = val_trackbar(frame)
        dst_pts = np.array([(0, height), (0, 0), (width, 0), (width, height)], dtype=np.float32)

        points = draw_points(frame.copy(), src_pts)

        img_warper = ImageWarper(src_pts, dst_pts)

        warped = img_warper.warp_matrix(frame)

        preprocessed_img = img_filter.preprocess_img(warped)

        _, thresh_rgb_r = img_filter.img_threshold(preprocessed_img, 190, 0)

        hls = img_filter.convert_to_hsl(preprocessed_img)

        # sobel = img_filterapply_sobel(hls, 1)

        yellow_mask = img_filter.select_yellow_hls(preprocessed_img)

        masked_image = sum_all_binary(thresh_rgb_r, yellow_mask)

        resized_images = list(map(img_filter.resize_img, (thresh_rgb_r, yellow_mask, masked_image)))

        vert = np.concatenate(resized_images)

        try:
            # If we found lines previously, run the simplified line fitter
            if leftLine.detected is True and rightLine.detected is True:
                left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits_from_prev(masked_image, masked_image, leftLine, rightLine, ym_per_pix, xm_per_pix)
            else:
                # Run the warped, binary image from the pipeline through the complex fitter
                left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(masked_image, ym_per_pix, xm_per_pix)
        except TypeError:
            pass

        # Add these fits to the line classes
        leftLine.add_new_fit(left_fit, left_fit_m)
        rightLine.add_new_fit(right_fit, left_fit_m)

        # get radius and center distance
        curve_rad = combine_radii(leftLine, rightLine)
        head, center_distance_m, center_distance_px = get_center_dist(leftLine, rightLine, xm_per_pix)

        # create the final image
        result = create_final_image(frame, masked_image, leftLine, rightLine, img_warper)
        
        # add the text to the image
        result = add_image_text(result, curve_rad, head, center_distance_m, center_distance_px)

        # cv2.imshow('perspective', warped)
        # cv2.imshow('points', points)

        cv2.imshow('vert', vert)

        cv2.imshow('lanes_image', img_filter.resize_img(out_img))
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
