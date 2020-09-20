package com.marjan.camera1;
import org.opencv.core.Mat;

final class TrainValues {

    private final Mat labels;
    private final Mat images;

    public TrainValues(Mat labels, Mat images) {
        this.labels = labels;
        this.images = images;
    }

    public Mat getLabels() {
        return labels;
    }

    public Mat getImages() {
        return images;
    }
}
