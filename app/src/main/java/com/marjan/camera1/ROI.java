package com.marjan.camera1;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ROI {

    public final Mat roi;
    public final Mat original;

    public ROI(Mat roi, Mat org) {
        this.roi = roi;
        this.original = org;
    }

    public Mat getRoi() {
        return roi;
    }

    public Mat getOriginal() {
        return original;
    }
}
