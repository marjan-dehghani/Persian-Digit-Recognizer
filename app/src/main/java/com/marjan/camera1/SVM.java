package com.marjan.camera1;

import android.content.Context;

import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.core.*;

public class SVM {

    CvSVMParams params = new CvSVMParams();
    CvSVM svm = new CvSVM();

    public Boolean train(Context context) {

        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_C(90);
        params.set_gamma(2);

        Mat varidx = new Mat();
        Mat sampleidx = new Mat();

        TrainData td = new TrainData();
        TrainValues tv = td.get(context);
        Mat images = tv.getImages();
        Mat labels = tv.getLabels();
        Boolean train = svm.train(images,labels,varidx,sampleidx,params);
        return train;
    }

    public Float test(Context context, Mat sample){
        return svm.predict(sample);
    }

}
