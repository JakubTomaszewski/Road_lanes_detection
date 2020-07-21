import cv2


class ImageWarper:

    def __init__(self, source_pts, dest_pts):
        self.source_pts = source_pts
        self.dest_pts = dest_pts

    def warp_matrix(self, image):
        perspective = cv2.getPerspectiveTransform(self.source_pts, self.dest_pts)
        shape = image.shape[1], image.shape[0]

        return cv2.warpPerspective(image, perspective, shape, flags=cv2.INTER_NEAREST)  # INTER_LINEAR

    def inverse_warp_matrix(self, image):
        perspective = cv2.getPerspectiveTransform(self.dest_pts, self.source_pts)
        shape = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, perspective, shape, flags=cv2.INTER_NEAREST)
