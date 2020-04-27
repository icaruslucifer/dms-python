package com.lucifer.icarus.app_dms;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONObject;

import java.util.HashMap;


public class MainActivity extends AppCompatActivity {

    //创建一个SoundPool对象
    SoundPool soundPool;
    public HashMap<String, Double> warnningMap = new HashMap<>();

    public HashMap<String,Integer> warnningIntervalMap = new HashMap<>();

    private TextView warningTV ;
    private ImageView cameraIV;

    private boolean needPlaySound = false;
    private boolean needShowImage = false;

    private ZmqMessageTask mTask = new ZmqMessageTask();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        warnningIntervalMap.put("smoke",2);
        warnningIntervalMap.put("phone",2);
        warnningIntervalMap.put("glance",2);
        warnningIntervalMap.put("yield",2);
        warnningIntervalMap.put("yawn",2);
        warnningIntervalMap.put("face",3);


        soundPool = new SoundPool(1, AudioManager.STREAM_SYSTEM, 5);

        (findViewById(R.id.button_setting)).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                gotoDebug();
            }
        });

        warningTV = (TextView)findViewById(R.id.textView_warning);
        cameraIV = (ImageView)findViewById(R.id.imageView_camera);

        ((CheckBox)(findViewById(R.id.checkBox_sound))).setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                needPlaySound = isChecked;
            }
        });

        ((CheckBox)(findViewById(R.id.checkBox_image))).setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                needShowImage = isChecked;
            }
        });

        mTask.execute();
    }


    public void  gotoDebug(){
        Intent intent = new Intent(this, com.lucifer.icarus.app_dms.DebugActivity.class);
        startActivity(intent);
    }


    protected int getSoundId(String wType){
        if (wType.equals("glance")){
            return soundPool.load(this,R.raw.glance,1);
        }else if (wType.equals("smoke")){
            return soundPool.load(this,R.raw.smoke,1);
        }else if (wType.equals("face")){
            return soundPool.load(this,R.raw.face,1);
        }else if (wType.equals("phone")){
            return soundPool.load(this,R.raw.phone,1);
        }else if (wType.equals("yawn")){
            return soundPool.load(this,R.raw.yawn,1);
        }else if (wType.equals("yield")) {
            return soundPool.load(this, R.raw.yield, 1);
        }else{
            return soundPool.load(this,R.raw.face,1);
        }
    }

    private void handleWarning(String wType){
        if (needPlaySound){
            playSound(wType);
        }
        showWarning(wType);
    }

    private void showWarning(String wType){
        if (wType.equals("glance")){
            warningTV.setText("请勿左顾右盼");
        }else if (wType.equals("smoke")){
            warningTV.setText("请勿抽烟");
        }else if (wType.equals("face")){
            warningTV.setText("无人驾驶？");
        }else if (wType.equals("phone")){
            warningTV.setText("开车请不要打电话");
        }else if (wType.equals("yawn")){
            warningTV.setText("请注意休息");
        }else if (wType.equals("yield")) {
            warningTV.setText("请正视前方");
        }else{
            warningTV.setText("");
        }
    }

    private void showImage(String imageStr){
        if(needShowImage){
            Bitmap b = stringToBitmap(imageStr);
            cameraIV.setImageBitmap(b);
        }
    }

    private void  playSound(String wType){
        final int soundId = getSoundId(wType);//加载音源id
        soundPool.setOnLoadCompleteListener(new SoundPool.OnLoadCompleteListener() {
            @Override
            public void onLoadComplete(SoundPool soundPool, int sampleId, int status) {
                soundPool.play(soundId, 1, 1, 0, 0, 1);
            }
        });
    }

    private class ZmqMessageTask extends AsyncTask<String,Integer,String>{

        private String addr =  "tcp://" + com.lucifer.icarus.app_dms.GlobalObject.getInstance().getMqttIP() + ":5000";
        private String currentMsg = "";

        private String imageStr = "";

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected String doInBackground(String... strings) {
            org.zeromq.ZMQ.Context context = org.zeromq.ZMQ.context(1);

            org.zeromq.ZMQ.Socket socket = context.socket(org.zeromq.ZMQ.SUB);
            socket.connect(addr);

            String filter = "";
            socket.subscribe(filter.getBytes());

            while(true){
                String msg = socket.recvStr();
                Log.d("zero", msg);
                currentMsg = msg;
                publishProgress(1);
            }
            //return null;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
            try {
                JSONObject obj = new JSONObject(currentMsg);
                if (obj.has("warning")){
                    String wType = obj.getString("warning");
                    double timestamp = obj.getDouble("time");
                    if (warnningMap.containsKey(wType)){
                        if (timestamp - warnningMap.get(wType) >warnningIntervalMap.get(wType)){
                            handleWarning(wType);
                            warnningMap.put(wType,timestamp);
                        }
                    }else{
                        handleWarning(wType);
                        warnningMap.put(wType,timestamp);
                    }
                }else if (obj.has("dtype")){
                    imageStr = obj.getString("data");
                    showImage(imageStr);
                }

            }catch (Exception e) {
                Log.e("debug", e.toString());
            }
        }
    }


    public static Bitmap stringToBitmap(String string) {
        Bitmap bitmap = null;
        try {
            byte[] bitmapArray = Base64.decode(string, Base64.DEFAULT);
            bitmap = BitmapFactory.decodeByteArray(bitmapArray, 0, bitmapArray.length);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bitmap;
    }

}
