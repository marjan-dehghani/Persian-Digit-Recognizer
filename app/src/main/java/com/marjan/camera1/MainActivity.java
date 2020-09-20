package com.marjan.camera1;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.*;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import java.lang.*;
import java.lang.reflect.Field;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.filter2D;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private Boolean cameraIsRunning;
    private Preprocessor preprocessor = new Preprocessor();
    private static final String TAG = "MainActivity";
    private Mat mByte;
    private Size size = new Size(80,80);
    private Mat testRoi;
    private Button captureButton;
    private TextView label;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    private CameraBridgeViewBase mOpenCvCameraView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        captureButton = findViewById(R.id.capture);
        label = findViewById(R.id.label);
        label.setText("Scan number below!");
        getCameraPermission();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            cameraIsRunning = false;
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            cameraIsRunning = false;
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        cameraIsRunning = true;
        label.setText("Scan number below!");
        mByte = new Mat(height, width, CvType.CV_8UC4);
        testRoi = new Mat(80,80,CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        cameraIsRunning = false;
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mByte = inputFrame.rgba();
        ROI result = preprocessor.process(mByte);
        testRoi = result.getRoi();
        return result.getOriginal();
    }

    public void captureTapped(View view){

        if (cameraIsRunning){
            captureButton.setText("Retake");
            start();
            mOpenCvCameraView.disableView();
        }else{
            captureButton.setText("Capture");
            mOpenCvCameraView.enableView();
        }
    }

    public void start(){
        Descriptor hogDescriptor = new Descriptor();
        SVM svm = new SVM();
        Boolean b = svm.train(this);
        if (b){

            Imgproc.cvtColor(testRoi,testRoi,Imgproc.COLOR_BGR2GRAY);
            Mat hoged = hogDescriptor.getDescription(testRoi);
            testRoi.release();
            Mat testData = new Mat(1,hoged.rows(),CvType.CV_32FC1);
            for(int j = 0 ; j < hoged.rows() ; j++)
                {
                    if (hoged.get(j , 0) != null)
                    {
                        testData.put(0 , j , hoged.get(j , 0)[0]) ;
                    }
                    else {
                        break;
                    }
                }
            hoged.release();
            Float result = svm.test(this,testData);
            testData.release();
            Integer resultInt = Math.round(result);
            label.setText(resultInt.toString());
        }
    }

//    public void getWriteStoragePermission(){
//        int hasStorageWritePermission = ContextCompat.checkSelfPermission(this,Manifest.permission.WRITE_EXTERNAL_STORAGE);
//        if (hasStorageWritePermission == PackageManager.PERMISSION_GRANTED) {
//            getReadStoragePermission();
//        } else{
//            ActivityCompat.requestPermissions(this,new String[]{  Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
//        }
//    }
//
//    public void getReadStoragePermission(){
//        int hasStorageWritePermission = ContextCompat.checkSelfPermission(this,Manifest.permission.READ_EXTERNAL_STORAGE);
//        if (hasStorageWritePermission == PackageManager.PERMISSION_GRANTED) {
//            getCameraPermission();
//        } else{
//            ActivityCompat.requestPermissions(this,new String[]{  Manifest.permission.READ_EXTERNAL_STORAGE}, 2);
//        }
//    }

    public void getCameraPermission(){
        int hasCameraPermission = ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA);
        if (hasCameraPermission == PackageManager.PERMISSION_GRANTED){
            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
            captureButton.setVisibility(View.VISIBLE);
        }else{
            ActivityCompat.requestPermissions(MainActivity.this,new String[]{ Manifest.permission.CAMERA}, 3);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
//            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                getReadStoragePermission();
//            }else{
//                captureButton.setVisibility(View.INVISIBLE);
//                Toast.makeText(this, "Write Storage Permission Denied", Toast.LENGTH_SHORT).show();
//                label.setText("Application needs to write external storage to continue");
//            }
        }else if (requestCode == 2){
//            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                getCameraPermission();
//            }else{
//                captureButton.setVisibility(View.INVISIBLE);
//                Toast.makeText(this, "Write Storage Permission Denied", Toast.LENGTH_SHORT).show();
//                label.setText("Application needs to read external storage to continue");
//            }
        }else if (requestCode == 3){
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
                mOpenCvCameraView.setCvCameraViewListener(this);
                captureButton.setVisibility(View.VISIBLE);
                label.setText("Scan number below!");
            }else{
                label.setText("Application needs camera access to continue");
                Toast.makeText(this, "Camera Access Denied", Toast.LENGTH_SHORT).show();
                captureButton.setVisibility(View.INVISIBLE);
            }
        }
    }
}
