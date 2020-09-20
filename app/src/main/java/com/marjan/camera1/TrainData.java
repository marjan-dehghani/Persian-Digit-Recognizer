package com.marjan.camera1;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

public class TrainData {

    Size size = new Size(80,80);
    ArrayList<Integer> trainingLabels = new ArrayList<>();
    Descriptor hogDescriptor = new Descriptor();
    Preprocessor preprocessor = new Preprocessor();
    Integer trainDataColumns = 0;
    Integer number_of_imgs = 0;
    Mat trainData = new Mat();

    public void compute(Context context, Integer resourceId, Integer label){
        Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), resourceId);
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap,mat);
        Imgproc.cvtColor(mat,mat,Imgproc.COLOR_BGR2GRAY);
        Imgproc.resize(mat,mat,size);
        Mat hoged = hogDescriptor.getDescription(mat);
        mat.release();
        Mat temp = new Mat(1,trainDataColumns,CvType.CV_32FC1);
        for(int j = 0 ; j < trainDataColumns ; j++)
        {
            temp.put(0 , j , hoged.get(j , 0)[0]) ;
        }
        hoged.release();
        trainData.push_back(temp);
        trainingLabels.add(label);
    }


    public TrainValues get(Context context){
        Field[] allImageFields = R.drawable.class.getFields();

        List<Integer> zeroImageIds = new ArrayList<Integer>();
        List<Integer> oneImageIds = new ArrayList<Integer>();
        List<Integer> twoImageIds = new ArrayList<Integer>();
        List<Integer> threeImageIds = new ArrayList<Integer>();
        List<Integer> fourImageIds = new ArrayList<Integer>();
        List<Integer> fiveImageIds = new ArrayList<Integer>();
        List<Integer> sixImageIds = new ArrayList<Integer>();
        List<Integer> sevenImageIds = new ArrayList<Integer>();
        List<Integer> eightImageIds = new ArrayList<Integer>();
        List<Integer> nineImageIds = new ArrayList<Integer>();

        for (Field field : allImageFields) {
            try {
                if (field.getName().endsWith("0") && field.getName().startsWith("td")) {
                    zeroImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("1") && field.getName().startsWith("td")) {
                    oneImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("2") && field.getName().startsWith("td")) {
                    twoImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("3") && field.getName().startsWith("td")) {
                    threeImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("4") && field.getName().startsWith("td")) {
                    fourImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("5") && field.getName().startsWith("td")) {
                    fiveImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("6") && field.getName().startsWith("td")) {
                    sixImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("7") && field.getName().startsWith("td")) {
                    sevenImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("8") && field.getName().startsWith("td")) {
                    eightImageIds.add(field.getInt(null));
                } else if (field.getName().endsWith("9") && field.getName().startsWith("td")) {
                    nineImageIds.add(field.getInt(null));
                }
            }
            catch(Exception e){
                return null;
            }
        }

        number_of_imgs = zeroImageIds.size() + oneImageIds.size() + twoImageIds.size() + threeImageIds.size() + fourImageIds.size() + fiveImageIds.size() + sixImageIds.size() + sevenImageIds.size() + eightImageIds.size() + nineImageIds.size();

        Integer id = zeroImageIds.get(0);
        Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), id);
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap,mat);
        Imgproc.cvtColor(mat,mat,Imgproc.COLOR_BGR2GRAY);
        Imgproc.resize(mat,mat,size);
        Mat hoged = hogDescriptor.getDescription(mat);
        mat.release();
        trainDataColumns = hoged.rows();
        trainData = new Mat(0,trainDataColumns,CvType.CV_32FC1);


        for (Integer id_ : zeroImageIds){
            compute(context,id_,0);
        }

        for (Integer id_ : oneImageIds){
            compute(context,id_,1);
        }

        for (Integer id_ : twoImageIds){
            compute(context,id_,2);
        }

        for (Integer id_ : threeImageIds){
            compute(context,id_,3);
        }

        for (Integer id_ : fourImageIds){
            compute(context,id_,4);
        }

        for (Integer id_ : fiveImageIds){
            compute(context,id_,5);
        }

        for (Integer id_ : sixImageIds){
            compute(context,id_,6);
        }

        for (Integer id_ : sevenImageIds){
            compute(context,id_,7);
        }

        for (Integer id_ : eightImageIds){
            compute(context,id_,8);
        }

        for (Integer id_ : nineImageIds){
            compute(context,id_,9);
        }

        Mat labels = new Mat(trainingLabels.size(), 1, CvType.CV_32SC1);

        for (int i = 0; i < labels.rows(); i++) {
            int value = trainingLabels.get(i);
            labels.put(i,0,value);
        }

        printMat(labels);
        return new TrainValues(labels,trainData);
    }

    void printMat(Mat matrix)
    {

        for(int i = 0 ; i < matrix.rows() ; i++) {
            for (int j = 0; j < matrix.cols(); j++) {

                Log.i("Value["+i+"]"+"["+j+"]",matrix.get(i,j)[0] + "");
            }
        }

    }
}
