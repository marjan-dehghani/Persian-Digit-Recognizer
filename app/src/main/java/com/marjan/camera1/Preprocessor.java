package com.marjan.camera1;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class Preprocessor {

    private Size blurKernel = new Size(7,7);

    public ROI process(Mat srcColored) {
        Mat duplicate = new Mat();
        cvtColor(srcColored,duplicate,Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(duplicate, duplicate, blurKernel, 0);
        Imgproc.threshold(duplicate,duplicate,127,255,THRESH_BINARY_INV);
        Imgproc.dilate(duplicate,duplicate, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2)));
        List<MatOfPoint> cnts = new ArrayList<MatOfPoint>();
        Imgproc.findContours(duplicate,cnts,new Mat(),RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        duplicate.release();
        Mat roi = new Mat(80,80,CvType.CV_8UC4);
            Integer i = 0;
            for (MatOfPoint cnt : cnts){
                if (Imgproc.contourArea(cnt) < 80 || Imgproc.contourArea(cnt) > 700){
                    continue;
                }
                Rect rect = Imgproc.boundingRect(cnt);
                Integer widthOffset = (80 - rect.width) / 2;
                Integer heightOffset = (80 - rect.height) / 2;
                Integer newX = rect.x - widthOffset;
                Integer newY = rect.y - heightOffset;
                if (newX >=0 && newY>=0 && newX+80 <= srcColored.cols() && newY+80 <= srcColored.rows()) {
                    roi = srcColored.submat(newY, newY + 80, newX, newX + 80);
                }
                Scalar roiColor = new Scalar(0, 255, 0);
                Core.rectangle(srcColored, new Point(newX, newY), new Point(newX + 80, newY + 80), roiColor, 2);
                i = i + 1;

            }
            ROI result = new ROI(roi,srcColored);
            return result;
    }


}
