package com.lucifer.icarus.app_dms;

import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.content.Intent;

import java.util.LinkedList;
import java.util.List;

public class ICAApplication extends Application {
    private static ICAApplication instance;

    private Intent mqttIntent = null;

    //存储所有的activity
    private List<Activity> activityList = new LinkedList<Activity>();

    public static ICAApplication getInstance() {
        if (instance == null) {
            instance = new ICAApplication();
        }
        return instance;
    }

    public static Context getAppContext() {
        return instance.getApplicationContext();
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;

        //在这里未应用设置异常处理，然后程序才能获取到未处理的异常
//        CrashHandler crashHandler = CrashHandler.getInstance();
//        crashHandler.init(this);
    }

    //添加Activity到容器中
    public void addActivity(Activity activity) {
        activityList.add(activity);
    }

    //从容器中删除Activity
    public void removeActivity(Activity activity) {
        activityList.remove(activity);
    }

    //遍历所有Activity并finish
    public void exit() {
        for (Activity activity : activityList) {
            activity.finish();
        }
        System.exit(0);
    }
}
