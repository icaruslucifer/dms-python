package com.lucifer.icarus.app_dms;

import android.content.Context;
import android.content.SharedPreferences;

import com.lucifer.icarus.app_dms.ICAApplication;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class GlobalObject {

    public  final static String SETTING_PCW = "pcw";
    public  final static String SETTING_LDW = "ldw";
    public  final static String SETTING_FCW = "fcw";
    public  final static String SETTING_DMS = "dms";
    public  final static String SETTING_BSD = "bsd";


    public final static float FCW_TTC_NORMAL = 2.7f;
    public final static float FCW_TTC_EARLY = 2.0f;
    public final static float FCW_TTC_LATER = 3.5f;

    //多久没触摸屏幕，自动跳出avm
    public final static int  touch_interval_threshold = 30;




    public final static int GEAR_N = 11;
    public final static int GEAR_P = 12;
    public final static int GEAR_R = 14;
    public final static int GEAR_D = 13;
    public final static int GEAR_L = 15;

    public final static int LIGHT_LEFT = 21;
    public final static int LIGHT_RIGHT = 22;
    public final static int LIGHT_ALL = 23;
    public final static int LIGHT_OFF = 20;

    public final static int BRAKE_ON = 31;
    public final static int BRAKE_OFF = 32;


    //对程序的一些toast进行设置
    public  final static String ToastMqtt = "mqtt";
    public  final static String ToastVideo = "video";
    public HashMap<String,Boolean> toastMap = new HashMap<>();

    public HashMap<String,String>  carMap = new HashMap<>();
    public HashMap<String,Boolean> needAlarmMap = new HashMap<>();

    public List<String> alarms = new ArrayList<>();

    public float fcw_ttc = 2.7f;
    public boolean need_send_user_config = true;

    private String needAlarmPreferenceKey = "needAlarm";
    private String fcwAlarmPreferenceKey= "fcwAlarm";

    private String mqttSettingKey = "mqttsetting";

    private String toastSettingKey = "toastsetting";


    private static class SingletonHolder {
        static final GlobalObject INSTANCE = new GlobalObject();
    }
    public static GlobalObject getInstance(){
        return GlobalObject.SingletonHolder.INSTANCE;
    }


    private GlobalObject() {
        alarms.add(SETTING_BSD);
        alarms.add(SETTING_DMS);
        alarms.add(SETTING_FCW);
        alarms.add(SETTING_LDW);
        alarms.add(SETTING_PCW);
        getAlarmSetting();
        updateToastSetting();
    }
    public void getAlarmSetting(){
        SharedPreferences sp = ICAApplication.getInstance().getSharedPreferences(needAlarmPreferenceKey, Context.MODE_PRIVATE);

        needAlarmMap.clear();
        for(int i=0;i<alarms.size();i++){
            String alarmkey = alarms.get(i);
            needAlarmMap.put(alarmkey,sp.getBoolean(alarmkey,true));
        }
        fcw_ttc = sp.getFloat(fcwAlarmPreferenceKey,fcw_ttc);
    }

    public void saveAlarmSetting(){
        SharedPreferences sharedPreferences = ICAApplication.getInstance().
                getSharedPreferences(needAlarmPreferenceKey, Context.MODE_PRIVATE);
        //获取editor对象
        SharedPreferences.Editor editor = sharedPreferences.edit();//获取编辑器
        //存储键值对
        editor.putBoolean(SETTING_FCW, needAlarmMap.get(SETTING_FCW));
        editor.putBoolean(SETTING_PCW, needAlarmMap.get(SETTING_PCW));
        editor.putBoolean(SETTING_LDW, needAlarmMap.get(SETTING_LDW));
        editor.putBoolean(SETTING_BSD, needAlarmMap.get(SETTING_BSD));
        editor.putBoolean(SETTING_DMS, needAlarmMap.get(SETTING_DMS));

        editor.putFloat(fcwAlarmPreferenceKey,fcw_ttc);

        //提交
        editor.apply();//提交修改
        getAlarmSetting();
    }


    public boolean openDMSAlarm(){
        return needAlarmMap.get(SETTING_DMS);
    }


    //设置mqtt的host
    public String getMqttIP(){
        SharedPreferences sp = ICAApplication.getInstance().getSharedPreferences(mqttSettingKey, Context.MODE_PRIVATE);
        String mqttip = "192.168.0.12";
        mqttip = sp.getString("mqttip",mqttip);
        return mqttip;
    }

    public void setMqttIP(String mqttip){
        SharedPreferences sharedPreferences = ICAApplication.getInstance().
                getSharedPreferences(mqttSettingKey, Context.MODE_PRIVATE);
        //获取editor对象
        SharedPreferences.Editor editor = sharedPreferences.edit();//获取编辑器
        //存储键值对
        editor.putString("mqttip", mqttip);
        //提交
        editor.commit();//提交修改
    }


    public static String createCarMapValue(String s){
        String tmp = Long.toString(System.currentTimeMillis());
        tmp +="#";
        tmp += s;
        return tmp;
    }

    public static Long getCarKeyTime(String s){
        String [] arr = s.split("[#]");

        if(arr.length == 2){
            return Long.parseLong(arr[0]);
        }
        return  0L;
    }

    public static String getCarKeyValue(String s){
        String [] arr = s.split("[#]");

        if(arr.length == 2){
            return arr[1];
        }
        return  "";
    }


    public void updateToastSetting(){
        SharedPreferences sp = ICAApplication.getInstance().getSharedPreferences(toastSettingKey, Context.MODE_PRIVATE);
        toastMap.put(ToastMqtt,sp.getBoolean(ToastMqtt,true));
    }

    public void setToastSetting(String toastType,Boolean trueForShow){
        SharedPreferences sharedPreferences = ICAApplication.getInstance().
                getSharedPreferences(toastSettingKey, Context.MODE_PRIVATE);
        //获取editor对象
        SharedPreferences.Editor editor = sharedPreferences.edit();//获取编辑器
        //存储键值对
        editor.putBoolean(toastType, trueForShow);
        //提交
        editor.commit();//提交修改
    }
}