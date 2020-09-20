package com.marjan.camera1;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.objdetect.HOGDescriptor;

public class Descriptor {

        private org.opencv.core.Size winsize = new org.opencv.core.Size(80,80);
        private org.opencv.core.Size blocksize = new org.opencv.core.Size(40,40);
        private org.opencv.core.Size blockStride = new org.opencv.core.Size(10,10);
        private org.opencv.core.Size cellsize = new org.opencv.core.Size(20,20);
        private Integer nbins = 9;
        private Integer derivAper = 1;
        private Integer winSigma = -1;
        private Integer histogramNormType = 0;
        private Double L2HysThresh = 0.2;
        private Boolean gammalCorrection = true;
        private Integer nlevels = 64;
        private Integer useSignedGradients = 1;
        private HOGDescriptor hogDescriptor = new HOGDescriptor(winsize,blocksize,blockStride,cellsize,nbins,derivAper,winSigma,histogramNormType,L2HysThresh,gammalCorrection,nlevels);

    public MatOfFloat getDescription(Mat img){
        MatOfFloat output = new MatOfFloat();
        hogDescriptor.getDescriptorSize();
        hogDescriptor.compute(img,output);
        return output;
    }


}
